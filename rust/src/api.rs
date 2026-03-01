use std::{sync::Arc, time::Instant};

use anyhow::Result;
use axum::{
    extract::{Request, State},
    http::{HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::{net::TcpListener, signal};
use tracing::info;

use crate::{
    engine::OnnxNerEngine,
    types::{InputEntry, PredictionOutput},
};

#[derive(Clone)]
pub struct ApiState {
    pub engine: Arc<tokio::sync::Mutex<OnnxNerEngine>>,
    pub default_batch_size: usize,
    pub default_max_length: usize,
    pub backend: String,
    pub reference: String,
}

impl ApiState {
    pub fn new(
        engine: OnnxNerEngine,
        default_batch_size: usize,
        default_max_length: usize,
    ) -> Self {
        Self {
            backend: engine.backend().to_string(),
            reference: engine.reference().to_string(),
            engine: Arc::new(tokio::sync::Mutex::new(engine)),
            default_batch_size,
            default_max_length,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct PredictItem {
    #[serde(default)]
    pub id: Option<Value>,
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub struct PredictRequest {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub texts: Option<Vec<String>>,
    #[serde(default)]
    pub items: Option<Vec<PredictItem>>,
    #[serde(default)]
    pub batch_size: Option<usize>,
    #[serde(default)]
    pub max_length: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct PredictResponse {
    pub count: usize,
    pub backend: String,
    pub results: Vec<PredictionOutput>,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    backend: String,
    reference: String,
}

#[derive(Debug)]
enum ApiError {
    BadRequest(String),
    Internal(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        match self {
            ApiError::BadRequest(message) => {
                (StatusCode::BAD_REQUEST, Json(json!({ "detail": message }))).into_response()
            }
            ApiError::Internal(message) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "detail": message })),
            )
                .into_response(),
        }
    }
}

pub async fn run_server(state: ApiState, host: &str, port: u16) -> Result<()> {
    let app = build_router(state);
    let bind_addr = format!("{host}:{port}");
    let listener = TcpListener::bind(&bind_addr).await?;
    info!("listening on http://{bind_addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

fn build_router(state: ApiState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/predict", post(predict_handler))
        .with_state(state)
        .layer(middleware::from_fn(process_time_middleware))
}

async fn health_handler(State(state): State<ApiState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        backend: state.backend,
        reference: state.reference,
    })
}

async fn predict_handler(
    State(state): State<ApiState>,
    Json(request): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, ApiError> {
    let entries = collect_inputs(&request)?;

    let batch_size = request.batch_size.unwrap_or(state.default_batch_size);
    if !(1..=1024).contains(&batch_size) {
        return Err(ApiError::BadRequest(
            "batch_size must be within [1, 1024]".to_string(),
        ));
    }

    let max_length = request.max_length.unwrap_or(state.default_max_length);
    if !(8..=4096).contains(&max_length) {
        return Err(ApiError::BadRequest(
            "max_length must be within [8, 4096]".to_string(),
        ));
    }

    let texts: Vec<String> = entries.iter().map(|x| x.text.clone()).collect();
    let mut engine = state.engine.lock().await;
    let rows = engine
        .predict_texts(&texts, batch_size, max_length)
        .map_err(|e| ApiError::Internal(e.to_string()))?;

    let mut results = Vec::with_capacity(rows.len());
    for (entry, row) in entries.into_iter().zip(rows.into_iter()) {
        results.push(PredictionOutput {
            row,
            id: entry.item_id,
        });
    }

    Ok(Json(PredictResponse {
        count: results.len(),
        backend: state.backend,
        results,
    }))
}

async fn process_time_middleware(request: Request, next: Next) -> Response {
    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let start = Instant::now();
    let mut response = next.run(request).await;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    if let Ok(value) = HeaderValue::from_str(&format!("{elapsed_ms:.2}")) {
        response.headers_mut().insert("X-Process-Time-Ms", value);
    }

    info!(
        "{} {} status={} duration_ms={:.2}",
        method,
        path,
        response.status().as_u16(),
        elapsed_ms
    );
    response
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};
        if let Ok(mut term) = signal(SignalKind::terminate()) {
            let _ = term.recv().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    info!("shutdown signal received");
}

fn collect_inputs(request: &PredictRequest) -> Result<Vec<InputEntry>, ApiError> {
    let mut entries = Vec::new();

    if let Some(text) = request.text.as_ref() {
        entries.push(InputEntry {
            text: normalize_text(text)?,
            item_id: None,
        });
    }

    if let Some(texts) = request.texts.as_ref() {
        for text in texts {
            entries.push(InputEntry {
                text: normalize_text(text)?,
                item_id: None,
            });
        }
    }

    if let Some(items) = request.items.as_ref() {
        for item in items {
            entries.push(InputEntry {
                text: normalize_text(&item.text)?,
                item_id: item.id.clone(),
            });
        }
    }

    if entries.is_empty() {
        return Err(ApiError::BadRequest(
            "Provide at least one of: text, texts, items.".to_string(),
        ));
    }
    Ok(entries)
}

fn normalize_text(text: &str) -> Result<String, ApiError> {
    let value = text.trim();
    if value.is_empty() {
        return Err(ApiError::BadRequest(
            "Input text cannot be empty.".to_string(),
        ));
    }
    Ok(value.to_string())
}
