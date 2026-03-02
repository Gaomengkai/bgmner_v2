use std::{
    collections::{BTreeSet, HashSet},
    env, fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{bail, Context, Result};
use ndarray::Array2;
use ort::{session::Session, value::TensorRef};
use serde_json::Value;
use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
    TruncationParams, TruncationStrategy,
};

use crate::{
    bio::decode_entities,
    provider::{build_provider_dispatch_chain, resolve_provider_argument},
    types::PredictionRow,
};

struct EncodedBatch {
    words_batch: Vec<Vec<String>>,
    word_ids_batch: Vec<Vec<i32>>,
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    batch_size: usize,
    sequence_length: usize,
}

#[derive(Default)]
struct StageTotals {
    batches: usize,
    texts: usize,
    encode: Duration,
    tensor: Duration,
    infer: Duration,
    argmax: Duration,
    decode: Duration,
}

impl StageTotals {
    fn reset(&mut self, texts: usize) {
        *self = Self {
            texts,
            ..Self::default()
        };
    }

    fn total(&self) -> Duration {
        self.encode + self.tensor + self.infer + self.argmax + self.decode
    }

    fn to_summary_line(&self) -> String {
        let total = self.total().as_secs_f64() * 1000.0;
        if total <= 0.0 {
            return format!("[profile] texts={} batches={} total_ms=0.00", self.texts, self.batches);
        }
        format!(
            "[profile] texts={} batches={} total_ms={:.2} encode_ms={:.2} ({:.1}%) tensor_ms={:.2} ({:.1}%) infer_ms={:.2} ({:.1}%) argmax_ms={:.2} ({:.1}%) decode_ms={:.2} ({:.1}%)",
            self.texts,
            self.batches,
            total,
            self.encode.as_secs_f64() * 1000.0,
            self.encode.as_secs_f64() * 1000.0 * 100.0 / total,
            self.tensor.as_secs_f64() * 1000.0,
            self.tensor.as_secs_f64() * 1000.0 * 100.0 / total,
            self.infer.as_secs_f64() * 1000.0,
            self.infer.as_secs_f64() * 1000.0 * 100.0 / total,
            self.argmax.as_secs_f64() * 1000.0,
            self.argmax.as_secs_f64() * 1000.0 * 100.0 / total,
            self.decode.as_secs_f64() * 1000.0,
            self.decode.as_secs_f64() * 1000.0 * 100.0 / total,
        )
    }
}

pub struct OnnxNerEngine {
    tokenizer: Tokenizer,
    session: Session,
    id2label: Vec<String>,
    pad_id: u32,
    pad_token: String,
    reference: String,
    profile_stages: bool,
    stage_totals: StageTotals,
}

impl OnnxNerEngine {
    pub fn load(
        model_dir: &Path,
        onnx_path: &Path,
        provider: &str,
        dml_device_id: i32,
    ) -> Result<Self> {
        if !model_dir.exists() {
            bail!("Model dir not found: {}", model_dir.display());
        }
        if !onnx_path.exists() {
            bail!("ONNX file not found: {}", onnx_path.display());
        }

        initialize_ort_runtime()?;
        let provider_resolution = resolve_provider_argument(provider)?;
        let provider_dispatches =
            build_provider_dispatch_chain(&provider_resolution.chain, dml_device_id)?;

        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            bail!("Tokenizer not found: {}", tokenizer_path.display());
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
            .with_context(|| format!("failed to load tokenizer: {}", tokenizer_path.display()))?;

        let id2label = load_id2label(model_dir)?;
        let (pad_id, pad_token) = resolve_padding(&tokenizer);

        let session = Session::builder()
            .context("failed to create ONNX session builder")?
            .with_execution_providers(provider_dispatches)
            .with_context(|| {
                format!(
                    "failed to apply provider chain {:?} (available: {:?})",
                    provider_resolution.chain, provider_resolution.available
                )
            })?
            .commit_from_file(onnx_path)
            .with_context(|| format!("failed to load ONNX: {}", onnx_path.display()))?;

        let profile_stages = env_flag_enabled("BGMNER_PROFILE_STAGES");
        Ok(Self {
            tokenizer,
            session,
            id2label,
            pad_id,
            pad_token,
            reference: format!(
                "{} @ {}",
                onnx_path.display(),
                provider_resolution.chain.join(",")
            ),
            profile_stages,
            stage_totals: StageTotals::default(),
        })
    }

    pub fn backend(&self) -> &'static str {
        "onnx"
    }

    pub fn reference(&self) -> &str {
        &self.reference
    }

    pub fn predict_texts(
        &mut self,
        texts: &[String],
        batch_size: usize,
        max_length: usize,
    ) -> Result<Vec<PredictionRow>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if batch_size == 0 {
            bail!("batch_size must be >= 1");
        }
        if max_length == 0 {
            bail!("max_length must be >= 1");
        }

        if self.profile_stages {
            self.stage_totals.reset(texts.len());
        }

        let mut all_rows = Vec::new();
        for chunk in texts.chunks(batch_size) {
            all_rows.extend(self.predict_batch(chunk, max_length)?);
        }
        if self.profile_stages {
            eprintln!("{}", self.stage_totals.to_summary_line());
        }
        Ok(all_rows)
    }

    fn predict_batch(&mut self, texts: &[String], max_length: usize) -> Result<Vec<PredictionRow>> {
        let encode_start = Instant::now();
        let encoded = self.encode_batch(texts, max_length)?;
        let encode_elapsed = encode_start.elapsed();
        if encoded.batch_size == 0 {
            return Ok(Vec::new());
        }

        let tensor_start = Instant::now();
        let input_ids = Array2::from_shape_vec(
            (encoded.batch_size, encoded.sequence_length),
            encoded.input_ids,
        )
        .context("failed to build input_ids tensor")?;
        let attention_mask = Array2::from_shape_vec(
            (encoded.batch_size, encoded.sequence_length),
            encoded.attention_mask,
        )
        .context("failed to build attention_mask tensor")?;

        let input_ids_tensor = TensorRef::from_array_view(input_ids.view())
            .context("failed to create input_ids tensor view")?;
        let attention_mask_tensor = TensorRef::from_array_view(attention_mask.view())
            .context("failed to create attention_mask tensor view")?;
        let tensor_elapsed = tensor_start.elapsed();

        let infer_start = Instant::now();
        let outputs = self
            .session
            .run(ort::inputs! {
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor
            })
            .context("ONNX inference failed")?;

        let logits_value = outputs.get("logits").unwrap_or(&outputs[0]);
        let (shape, logits) = logits_value
            .try_extract_tensor::<f32>()
            .context("failed to extract logits data")?;
        if shape.len() != 3 {
            bail!("expected logits rank=3, got rank={}", shape.len());
        }

        let logits_batch = dim_to_usize(shape[0], "batch")?;
        let logits_seq = dim_to_usize(shape[1], "sequence")?;
        let num_labels = dim_to_usize(shape[2], "num_labels")?;
        if num_labels == 0 {
            bail!("num_labels cannot be zero");
        }
        if logits_batch != encoded.batch_size || logits_seq != encoded.sequence_length {
            bail!(
                "logits/input shape mismatch: logits=({}, {}), inputs=({}, {})",
                logits_batch,
                logits_seq,
                encoded.batch_size,
                encoded.sequence_length
            );
        }
        let infer_elapsed = infer_start.elapsed();

        let argmax_start = Instant::now();
        let mut token_pred_ids = vec![vec![0usize; logits_seq]; logits_batch];
        for (batch_idx, sample_preds) in token_pred_ids.iter_mut().enumerate().take(logits_batch) {
            for (token_idx, label_slot) in sample_preds.iter_mut().enumerate().take(logits_seq) {
                let base = (batch_idx * logits_seq + token_idx) * num_labels;
                let row = &logits[base..base + num_labels];
                let mut best_idx = 0usize;
                let mut best_val = f32::NEG_INFINITY;
                for (label_idx, val) in row.iter().enumerate() {
                    if *val > best_val {
                        best_val = *val;
                        best_idx = label_idx;
                    }
                }
                *label_slot = best_idx;
            }
        }
        let argmax_elapsed = argmax_start.elapsed();

        let decode_start = Instant::now();
        let mut rows = Vec::with_capacity(logits_batch);
        for idx in 0..logits_batch {
            let words = &encoded.words_batch[idx];
            let word_ids = &encoded.word_ids_batch[idx];
            let word_tags = token_ids_to_word_tags(&token_pred_ids[idx], word_ids, &self.id2label);
            let truncated_words: Vec<String> =
                words.iter().take(word_tags.len()).cloned().collect();
            let entities = decode_entities(&truncated_words, &word_tags)?;

            rows.push(PredictionRow {
                text: texts[idx].clone(),
                truncated_text: truncated_words.concat(),
                entities,
                pred_labels: word_tags,
            });
        }
        let decode_elapsed = decode_start.elapsed();

        if self.profile_stages {
            self.stage_totals.batches += 1;
            self.stage_totals.encode += encode_elapsed;
            self.stage_totals.tensor += tensor_elapsed;
            self.stage_totals.infer += infer_elapsed;
            self.stage_totals.argmax += argmax_elapsed;
            self.stage_totals.decode += decode_elapsed;
        }

        Ok(rows)
    }

    fn encode_batch(&self, texts: &[String], max_length: usize) -> Result<EncodedBatch> {
        let mut tokenizer = self.tokenizer.clone();
        tokenizer
            .with_truncation(Some(TruncationParams {
                direction: TruncationDirection::Right,
                max_length,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
            }))
            .map_err(|e| anyhow::anyhow!(e.to_string()))
            .context("failed to configure truncation")?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: self.pad_id,
            pad_type_id: 0,
            pad_token: self.pad_token.clone(),
        }));

        let words_batch: Vec<Vec<String>> = texts.iter().map(|x| text_to_words(x)).collect();
        let encodings = tokenizer
            .encode_batch(words_batch.clone(), true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
            .context("tokenizer encode_batch failed")?;

        let batch_size = encodings.len();
        let sequence_length = encodings
            .first()
            .map(|x| x.get_ids().len())
            .unwrap_or(0usize);
        let mut input_ids = Vec::with_capacity(batch_size * sequence_length);
        let mut attention_mask = Vec::with_capacity(batch_size * sequence_length);
        let mut word_ids_batch = Vec::with_capacity(batch_size);

        for encoding in &encodings {
            if encoding.get_ids().len() != sequence_length {
                bail!("tokenizer returned inconsistent sequence length inside batch");
            }
            input_ids.extend(encoding.get_ids().iter().map(|x| i64::from(*x)));
            attention_mask.extend(encoding.get_attention_mask().iter().map(|x| i64::from(*x)));
            word_ids_batch.push(
                encoding
                    .get_word_ids()
                    .iter()
                    .map(|x| x.map(|v| v as i32).unwrap_or(-1))
                    .collect(),
            );
        }

        Ok(EncodedBatch {
            words_batch,
            word_ids_batch,
            input_ids,
            attention_mask,
            batch_size,
            sequence_length,
        })
    }
}

fn text_to_words(text: &str) -> Vec<String> {
    text.chars().map(|x| x.to_string()).collect()
}

fn token_ids_to_word_tags(
    token_label_ids: &[usize],
    word_ids: &[i32],
    id2label: &[String],
) -> Vec<String> {
    let max_word_id = word_ids
        .iter()
        .copied()
        .filter(|x| *x >= 0)
        .max()
        .unwrap_or(-1);
    if max_word_id < 0 {
        return Vec::new();
    }

    let mut word_tags = vec!["O".to_string(); (max_word_id as usize) + 1];
    let mut seen = HashSet::<i32>::new();
    for (token_idx, word_id) in word_ids.iter().copied().enumerate() {
        if word_id < 0 || seen.contains(&word_id) {
            continue;
        }
        if token_idx >= token_label_ids.len() {
            break;
        }

        seen.insert(word_id);
        let label = id2label
            .get(token_label_ids[token_idx])
            .cloned()
            .unwrap_or_else(|| "O".to_string());
        word_tags[word_id as usize] = label;
    }
    word_tags
}

fn load_id2label(model_dir: &Path) -> Result<Vec<String>> {
    let config_path = model_dir.join("config.json");
    let raw = fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read config: {}", config_path.display()))?;
    let value: Value = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse config json: {}", config_path.display()))?;

    let mut mapping = std::collections::BTreeMap::<usize, String>::new();

    if let Some(obj) = value.get("id2label").and_then(Value::as_object) {
        for (k, v) in obj {
            let id = k
                .parse::<usize>()
                .with_context(|| format!("invalid id2label key: {k}"))?;
            let label = v
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("id2label[{k}] must be string"))?;
            mapping.insert(id, label.to_string());
        }
    } else if let Some(obj) = value.get("label2id").and_then(Value::as_object) {
        for (label, v) in obj {
            let id = if let Some(x) = v.as_i64() {
                usize::try_from(x)
                    .with_context(|| format!("negative label2id value for label {label}"))?
            } else if let Some(x) = v.as_u64() {
                usize::try_from(x)
                    .with_context(|| format!("label2id too large for label {label}"))?
            } else {
                bail!("label2id[{label}] must be number");
            };
            mapping.insert(id, label.clone());
        }
    } else {
        bail!("config.json missing id2label/label2id");
    }

    let max_id = mapping.keys().copied().max().unwrap_or(0usize);
    let mut labels = vec!["O".to_string(); max_id + 1];
    for (idx, label) in mapping {
        labels[idx] = label;
    }
    Ok(labels)
}

fn resolve_padding(tokenizer: &Tokenizer) -> (u32, String) {
    if let Some(padding) = tokenizer.get_padding() {
        return (padding.pad_id, padding.pad_token.clone());
    }

    let pad_id = tokenizer
        .token_to_id("<pad>")
        .or_else(|| tokenizer.token_to_id("[PAD]"))
        .or_else(|| tokenizer.token_to_id("<|pad|>"))
        .unwrap_or(0u32);
    let pad_token = tokenizer
        .id_to_token(pad_id)
        .unwrap_or_else(|| "<pad>".to_string());
    (pad_id, pad_token)
}

fn env_flag_enabled(name: &str) -> bool {
    let Some(value) = env::var_os(name) else {
        return false;
    };
    let text = value.to_string_lossy().trim().to_ascii_lowercase();
    matches!(text.as_str(), "1" | "true" | "yes" | "on")
}

fn dim_to_usize(dim: i64, name: &str) -> Result<usize> {
    if dim < 0 {
        bail!("tensor dimension {name} is negative: {dim}");
    }
    usize::try_from(dim).with_context(|| format!("tensor dimension {name} too large: {dim}"))
}

fn initialize_ort_runtime() -> Result<()> {
    maybe_preload_directml_from_exe_dir()?;

    if let Some(raw_path) = env::var_os("ORT_DYLIB_PATH") {
        let raw_text = raw_path.to_string_lossy();
        let trimmed = raw_text.trim();
        if trimmed.is_empty() {
            // Treat empty env var as unset and fall back to automatic discovery.
            env::remove_var("ORT_DYLIB_PATH");
        } else {
            let path = PathBuf::from(trimmed);
            if !path.exists() {
                bail!(
                    "ORT_DYLIB_PATH is set but file does not exist: {}",
                    path.display()
                );
            }
            let builder = ort::init_from(&path)
                .with_context(|| format!("failed to load ORT from {}", path.display()))?;
            let _ = builder.with_name("bgmner-rs").commit();
            return Ok(());
        }
    }

    let mut attempts = Vec::<String>::new();
    for candidate in collect_ort_dylib_candidates() {
        if !candidate.exists() {
            continue;
        }
        match ort::init_from(&candidate) {
            Ok(builder) => {
                let _ = builder.with_name("bgmner-rs").commit();
                env::set_var("ORT_DYLIB_PATH", &candidate);
                return Ok(());
            }
            Err(err) => attempts.push(format!("{} ({err})", candidate.display())),
        }
    }

    let tried = if attempts.is_empty() {
        "none".to_string()
    } else {
        attempts.join(" | ")
    };
    bail!(
        "unable to load ONNX Runtime dylib. Set ORT_DYLIB_PATH to your onnxruntime library path (e.g. <conda-env>/Lib/site-packages/onnxruntime/capi/onnxruntime.dll). Tried: {tried}"
    );
}

fn maybe_preload_directml_from_exe_dir() -> Result<()> {
    if !cfg!(target_os = "windows") {
        return Ok(());
    }
    let Ok(exe) = env::current_exe() else {
        return Ok(());
    };
    let Some(dir) = exe.parent() else {
        return Ok(());
    };
    let dml_path = dir.join("DirectML.dll");
    if !dml_path.exists() {
        return Ok(());
    }
    ort::util::preload_dylib(&dml_path)
        .with_context(|| format!("failed to preload DirectML runtime: {}", dml_path.display()))?;
    Ok(())
}

fn collect_ort_dylib_candidates() -> Vec<PathBuf> {
    let lib_name = if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    };

    let mut ordered = Vec::<PathBuf>::new();
    let mut seen = BTreeSet::<PathBuf>::new();

    if cfg!(target_os = "windows") {
        if let Ok(exe) = env::current_exe() {
            if let Some(dir) = exe.parent() {
                // Highest priority on Windows: dll next to executable.
                push_unique_candidate(&mut ordered, &mut seen, dir.join(lib_name));
            }
        }
    }

    if let Some(prefix) = env::var_os("CONDA_PREFIX") {
        push_env_root_candidates(
            &mut ordered,
            &mut seen,
            &PathBuf::from(prefix),
            lib_name,
            true,
        );
    }
    if let Some(venv) = env::var_os("VIRTUAL_ENV") {
        push_env_root_candidates(
            &mut ordered,
            &mut seen,
            &PathBuf::from(venv),
            lib_name,
            true,
        );
    }

    if let Some(raw_path) = env::var_os("PATH") {
        for entry in env::split_paths(&raw_path) {
            // Lowest priority fallback: dynamic loader search path.
            push_unique_candidate(&mut ordered, &mut seen, entry.join(lib_name));
            push_env_root_candidates(&mut ordered, &mut seen, &entry, lib_name, true);
            if let Some(parent) = entry.parent() {
                push_env_root_candidates(&mut ordered, &mut seen, parent, lib_name, true);
            }
        }
    }

    if !cfg!(target_os = "windows") {
        if let Ok(exe) = env::current_exe() {
            if let Some(dir) = exe.parent() {
                push_unique_candidate(&mut ordered, &mut seen, dir.join(lib_name));
            }
        }
    }
    ordered
}

fn push_unique_candidate(ordered: &mut Vec<PathBuf>, seen: &mut BTreeSet<PathBuf>, path: PathBuf) {
    let canonical = fs::canonicalize(&path).unwrap_or(path);
    if seen.insert(canonical.clone()) {
        ordered.push(canonical);
    }
}

fn push_env_root_candidates(
    ordered: &mut Vec<PathBuf>,
    seen: &mut BTreeSet<PathBuf>,
    root: &Path,
    lib_name: &str,
    enabled: bool,
) {
    if !enabled {
        return;
    }
    push_unique_candidate(
        ordered,
        seen,
        root.join("Lib")
            .join("site-packages")
            .join("onnxruntime")
            .join("capi")
            .join(lib_name),
    );
    push_unique_candidate(
        ordered,
        seen,
        root.join("lib")
            .join("python3.12")
            .join("site-packages")
            .join("onnxruntime")
            .join("capi")
            .join(lib_name),
    );
    push_unique_candidate(ordered, seen, root.join("Lib").join(lib_name));
    push_unique_candidate(ordered, seen, root.join("lib").join(lib_name));
    push_unique_candidate(ordered, seen, root.join(lib_name));
}
