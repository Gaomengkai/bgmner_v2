use anyhow::{bail, Result};
use ort::ep::{self, ExecutionProvider, ExecutionProviderDispatch};

pub const AUTO_PROVIDER: &str = "auto";
pub const CPU_EXECUTION_PROVIDER: &str = "CPUExecutionProvider";
pub const COREML_EXECUTION_PROVIDER: &str = "CoreMLExecutionProvider";
pub const DML_EXECUTION_PROVIDER: &str = "DmlExecutionProvider";
pub const CUDA_EXECUTION_PROVIDER: &str = "CUDAExecutionProvider";
pub const ROCM_EXECUTION_PROVIDER: &str = "ROCMExecutionProvider";

#[derive(Debug, Clone)]
pub struct ProviderResolution {
    pub chain: Vec<String>,
    pub available: Vec<String>,
}

pub fn resolve_provider_argument(provider: &str) -> Result<ProviderResolution> {
    let available = available_known_providers()?;
    if available.is_empty() {
        bail!("onnxruntime returned no available execution providers.");
    }

    let requested = parse_provider_argument(provider)?;
    if requested.len() == 1 && requested[0].eq_ignore_ascii_case(AUTO_PROVIDER) {
        let mut resolved = default_provider_priority()
            .into_iter()
            .filter(|name| available.iter().any(|x| x == name))
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        if resolved.is_empty() {
            resolved.push(available[0].clone());
        }
        return Ok(ProviderResolution {
            chain: resolved,
            available,
        });
    }

    let requested = dedupe_ordered(requested);
    let missing = requested
        .iter()
        .filter(|name| !available.iter().any(|item| item == *name))
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        bail!(
            "Provider(s) {:?} not available. Requested: {:?}. Available: {:?}.",
            missing,
            requested,
            available
        );
    }
    Ok(ProviderResolution {
        chain: requested,
        available,
    })
}

pub fn build_provider_dispatch_chain(
    provider_chain: &[String],
    dml_device_id: i32,
) -> Result<Vec<ExecutionProviderDispatch>> {
    let mut dispatches = Vec::with_capacity(provider_chain.len());
    for (idx, name) in provider_chain.iter().enumerate() {
        let dispatch = provider_name_to_dispatch(name, dml_device_id)?;
        let dispatch = if provider_chain.len() > 1 && idx + 1 < provider_chain.len() {
            // Allow fallback to the next provider in chain.
            dispatch.fail_silently()
        } else {
            dispatch.error_on_failure()
        };
        dispatches.push(dispatch);
    }
    Ok(dispatches)
}

fn parse_provider_argument(provider: &str) -> Result<Vec<String>> {
    let value = provider.trim();
    if value.is_empty() {
        return Ok(vec![AUTO_PROVIDER.to_string()]);
    }

    let mut result = Vec::<String>::new();
    for raw in value.split(',') {
        let item = raw.trim();
        if item.is_empty() {
            continue;
        }
        if item.eq_ignore_ascii_case(AUTO_PROVIDER) {
            result.push(AUTO_PROVIDER.to_string());
            continue;
        }

        let canonical = canonical_provider_name(item).ok_or_else(|| {
            anyhow::anyhow!(
                "Unsupported provider '{item}'. Supported aliases: cpu/coreml/dml/cuda/rocm and official names: {CPU_EXECUTION_PROVIDER}, {COREML_EXECUTION_PROVIDER}, {DML_EXECUTION_PROVIDER}, {CUDA_EXECUTION_PROVIDER}, {ROCM_EXECUTION_PROVIDER}"
            )
        })?;
        result.push(canonical.to_string());
    }

    if result.is_empty() {
        return Ok(vec![AUTO_PROVIDER.to_string()]);
    }
    Ok(result)
}

fn canonical_provider_name(name: &str) -> Option<&'static str> {
    let lower = name.to_ascii_lowercase();
    match lower.as_str() {
        "cpu" | "cpuexecutionprovider" => Some(CPU_EXECUTION_PROVIDER),
        "coreml" | "coremlexecutionprovider" => Some(COREML_EXECUTION_PROVIDER),
        "dml" | "dmlexecutionprovider" => Some(DML_EXECUTION_PROVIDER),
        "cuda" | "cudaexecutionprovider" => Some(CUDA_EXECUTION_PROVIDER),
        "rocm" | "rocmexecutionprovider" => Some(ROCM_EXECUTION_PROVIDER),
        _ => None,
    }
}

fn provider_name_to_dispatch(name: &str, dml_device_id: i32) -> Result<ExecutionProviderDispatch> {
    match name {
        CPU_EXECUTION_PROVIDER => Ok(ep::CPU::default().with_arena_allocator(true).build()),
        COREML_EXECUTION_PROVIDER => Ok(ep::CoreML::default().build()),
        DML_EXECUTION_PROVIDER => Ok(ep::DirectML::default()
            .with_device_id(dml_device_id)
            .build()),
        CUDA_EXECUTION_PROVIDER => Ok(ep::CUDA::default().build()),
        ROCM_EXECUTION_PROVIDER => Ok(ep::ROCm::default().build()),
        _ => bail!(
            "Unsupported provider name '{name}'. Supported: {CPU_EXECUTION_PROVIDER}, {COREML_EXECUTION_PROVIDER}, {DML_EXECUTION_PROVIDER}, {CUDA_EXECUTION_PROVIDER}, {ROCM_EXECUTION_PROVIDER}"
        ),
    }
}

fn available_known_providers() -> Result<Vec<String>> {
    let mut providers = Vec::<String>::new();
    if ep::CPU::default().is_available()? {
        providers.push(CPU_EXECUTION_PROVIDER.to_string());
    }
    if ep::CoreML::default().is_available()? {
        providers.push(COREML_EXECUTION_PROVIDER.to_string());
    }
    if ep::DirectML::default().is_available()? {
        providers.push(DML_EXECUTION_PROVIDER.to_string());
    }
    if ep::CUDA::default().is_available()? {
        providers.push(CUDA_EXECUTION_PROVIDER.to_string());
    }
    if ep::ROCm::default().is_available()? {
        providers.push(ROCM_EXECUTION_PROVIDER.to_string());
    }
    Ok(providers)
}

fn default_provider_priority() -> Vec<&'static str> {
    if cfg!(target_os = "macos") && (cfg!(target_arch = "aarch64")) {
        return vec![COREML_EXECUTION_PROVIDER, CPU_EXECUTION_PROVIDER];
    }
    vec![CPU_EXECUTION_PROVIDER]
}

fn dedupe_ordered(values: Vec<String>) -> Vec<String> {
    let mut result = Vec::<String>::new();
    for value in values {
        if !result.contains(&value) {
            result.push(value);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::{
        canonical_provider_name, dedupe_ordered, CPU_EXECUTION_PROVIDER, DML_EXECUTION_PROVIDER,
    };

    #[test]
    fn canonicalizes_alias_and_official_name() {
        assert_eq!(canonical_provider_name("dml"), Some(DML_EXECUTION_PROVIDER));
        assert_eq!(
            canonical_provider_name("DmlExecutionProvider"),
            Some(DML_EXECUTION_PROVIDER)
        );
        assert_eq!(canonical_provider_name("cpu"), Some(CPU_EXECUTION_PROVIDER));
        assert_eq!(
            canonical_provider_name("CPUExecutionProvider"),
            Some(CPU_EXECUTION_PROVIDER)
        );
    }

    #[test]
    fn dedupe_preserves_order() {
        let deduped = dedupe_ordered(vec![
            "A".to_string(),
            "B".to_string(),
            "A".to_string(),
            "C".to_string(),
        ]);
        assert_eq!(
            deduped,
            vec!["A".to_string(), "B".to_string(), "C".to_string()]
        );
    }
}
