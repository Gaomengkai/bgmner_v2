use serde::Serialize;
use serde_json::Value;

use crate::bio::EntityMap;

#[derive(Debug, Clone, Serialize)]
pub struct PredictionRow {
    pub text: String,
    pub truncated_text: String,
    pub entities: EntityMap,
    pub pred_labels: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct InputEntry {
    pub text: String,
    pub item_id: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PredictionOutput {
    #[serde(flatten)]
    pub row: PredictionRow,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
}
