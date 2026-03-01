use std::{
    fs::File,
    io::{BufRead, BufReader, Write},
    path::Path,
};

use anyhow::{bail, Context, Result};
use serde_json::Value;

use crate::types::{InputEntry, PredictionOutput};

pub fn load_batch_inputs(texts: &[String], input_file: Option<&Path>) -> Result<Vec<InputEntry>> {
    let mut entries = Vec::<InputEntry>::new();

    for text in texts {
        let value = normalize_text(text)?;
        entries.push(InputEntry {
            text: value,
            item_id: None,
        });
    }

    if let Some(path) = input_file {
        let file = File::open(path)
            .with_context(|| format!("failed to open input file: {}", path.display()))?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line = line.context("failed to read input line")?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if let Some((text, id)) = parse_input_line(trimmed)? {
                entries.push(InputEntry {
                    text: normalize_text(&text)?,
                    item_id: id,
                });
            } else {
                entries.push(InputEntry {
                    text: normalize_text(trimmed)?,
                    item_id: None,
                });
            }
        }
    }

    if entries.is_empty() {
        bail!("provide --text or --input-file with at least one non-empty sample");
    }
    Ok(entries)
}

pub fn write_batch_outputs(results: &[PredictionOutput], output_file: Option<&Path>) -> Result<()> {
    if let Some(path) = output_file {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).with_context(|| {
                    format!(
                        "failed to create output parent directory: {}",
                        parent.display()
                    )
                })?;
            }
        }
        let mut handle = File::create(path)
            .with_context(|| format!("failed to create output file: {}", path.display()))?;
        for row in results {
            writeln!(handle, "{}", serde_json::to_string(row)?).context("failed to write jsonl")?;
        }
        return Ok(());
    }

    if results.len() == 1 {
        println!("{}", serde_json::to_string_pretty(&results[0])?);
        return Ok(());
    }

    for row in results {
        println!("{}", serde_json::to_string(row)?);
    }
    Ok(())
}

fn parse_input_line(line: &str) -> Result<Option<(String, Option<Value>)>> {
    let value = match serde_json::from_str::<Value>(line) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };

    let object = match value.as_object() {
        Some(v) => v,
        None => return Ok(None),
    };

    let id = object.get("id").cloned();
    let Some(text_value) = object.get("text") else {
        return Ok(None);
    };

    if let Some(text) = text_value.as_str() {
        return Ok(Some((text.to_string(), id)));
    }

    if let Some(items) = text_value.as_array() {
        let mut text = String::new();
        for item in items {
            if let Some(s) = item.as_str() {
                text.push_str(s);
            } else {
                text.push_str(item.to_string().trim_matches('"'));
            }
        }
        return Ok(Some((text, id)));
    }

    Ok(None)
}

fn normalize_text(text: &str) -> Result<String> {
    let value = text.trim();
    if value.is_empty() {
        bail!("input text cannot be empty");
    }
    Ok(value.to_string())
}
