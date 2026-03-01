use std::collections::BTreeMap;

use anyhow::{bail, Result};

pub type EntityMap = BTreeMap<String, Vec<(String, usize, usize)>>;

pub fn normalize_bio_tags(tags: &[String]) -> Vec<String> {
    let mut normalized = Vec::with_capacity(tags.len());
    let mut active_type: Option<&str> = None;

    for tag in tags {
        if tag == "O" {
            normalized.push("O".to_string());
            active_type = None;
            continue;
        }

        let mut parts = tag.splitn(2, '-');
        let prefix = parts.next().unwrap_or_default();
        let label_type = match parts.next() {
            Some(x) if !x.is_empty() => x,
            _ => {
                normalized.push("O".to_string());
                active_type = None;
                continue;
            }
        };

        match prefix {
            "B" => {
                normalized.push(tag.clone());
                active_type = Some(label_type);
            }
            "I" => {
                if active_type == Some(label_type) {
                    normalized.push(tag.clone());
                } else {
                    normalized.push(format!("B-{label_type}"));
                }
                active_type = Some(label_type);
            }
            _ => {
                normalized.push("O".to_string());
                active_type = None;
            }
        }
    }

    normalized
}

pub fn decode_entities(words: &[String], tags: &[String]) -> Result<EntityMap> {
    if words.len() != tags.len() {
        bail!(
            "words/tags length mismatch: {} vs {}",
            words.len(),
            tags.len()
        );
    }

    let normalized = normalize_bio_tags(tags);
    let mut entities: EntityMap = BTreeMap::new();

    let mut active_type: Option<String> = None;
    let mut start_idx = 0usize;

    for index in 0..=normalized.len() {
        let tag = if index < normalized.len() {
            normalized[index].as_str()
        } else {
            "O"
        };

        if tag == "O" {
            if let Some(label_type) = active_type.take() {
                let end_idx = index.saturating_sub(1);
                let text = words[start_idx..=end_idx].concat();
                entities
                    .entry(label_type)
                    .or_default()
                    .push((text, start_idx, end_idx));
            }
            continue;
        }

        let mut parts = tag.splitn(2, '-');
        let prefix = parts.next().unwrap_or_default();
        let label_type = parts.next().unwrap_or_default();

        if prefix == "B" {
            if let Some(prev_type) = active_type.take() {
                let end_idx = index.saturating_sub(1);
                let text = words[start_idx..=end_idx].concat();
                entities
                    .entry(prev_type)
                    .or_default()
                    .push((text, start_idx, end_idx));
            }
            active_type = Some(label_type.to_string());
            start_idx = index;
            continue;
        }

        if prefix == "I" && active_type.is_none() {
            active_type = Some(label_type.to_string());
            start_idx = index;
        }
    }

    Ok(entities)
}

#[cfg(test)]
mod tests {
    use super::{decode_entities, normalize_bio_tags};

    #[test]
    fn normalize_rewrites_invalid_i_prefix() {
        let tags = vec!["I-CT".to_string(), "I-CT".to_string(), "O".to_string()];
        let normalized = normalize_bio_tags(&tags);
        assert_eq!(normalized, vec!["B-CT", "I-CT", "O"]);
    }

    #[test]
    fn decode_builds_entity_spans() {
        let words = vec!["你".to_string(), "好".to_string(), "啊".to_string()];
        let tags = vec!["B-X".to_string(), "I-X".to_string(), "O".to_string()];
        let entities = decode_entities(&words, &tags).expect("decode should succeed");
        let spans = entities.get("X").expect("missing label X");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0], ("你好".to_string(), 0, 1));
    }
}
