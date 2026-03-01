use std::path::PathBuf;

use anyhow::Result;
use bgmner_rs::{
    api::{run_server, ApiState},
    batch::{load_batch_inputs, write_batch_outputs},
    engine::OnnxNerEngine,
    types::PredictionOutput,
};
use clap::{Args, Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "bgmner-rs",
    version,
    about = "Rust ONNX NER (Web API + batch inference)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Serve(ServeArgs),
    Batch(BatchArgs),
}

#[derive(Debug, Args)]
struct ServeArgs {
    #[arg(long, required = true)]
    model_dir: PathBuf,
    #[arg(long, required = true)]
    onnx_path: PathBuf,
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    #[arg(long, default_value_t = 32)]
    batch_size: usize,
    #[arg(long, default_value_t = 256)]
    max_length: usize,
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Debug, Args)]
struct BatchArgs {
    #[arg(long, required = true)]
    model_dir: PathBuf,
    #[arg(long, required = true)]
    onnx_path: PathBuf,
    #[arg(long)]
    text: Vec<String>,
    #[arg(long)]
    input_file: Option<PathBuf>,
    #[arg(long)]
    output_file: Option<PathBuf>,
    #[arg(long, default_value_t = 32)]
    batch_size: usize,
    #[arg(long, default_value_t = 256)]
    max_length: usize,
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Serve(args) => run_serve(args).await,
        Command::Batch(args) => run_batch(args),
    }
}

async fn run_serve(args: ServeArgs) -> Result<()> {
    init_logging(&args.log_level);
    let engine = OnnxNerEngine::load(&args.model_dir, &args.onnx_path)?;
    let state = ApiState::new(engine, args.batch_size, args.max_length);
    run_server(state, &args.host, args.port).await
}

fn run_batch(args: BatchArgs) -> Result<()> {
    init_logging(&args.log_level);
    let mut engine = OnnxNerEngine::load(&args.model_dir, &args.onnx_path)?;

    let inputs = load_batch_inputs(&args.text, args.input_file.as_deref())?;
    let texts: Vec<String> = inputs.iter().map(|x| x.text.clone()).collect();
    let rows = engine.predict_texts(&texts, args.batch_size, args.max_length)?;

    let outputs: Vec<PredictionOutput> = inputs
        .into_iter()
        .zip(rows)
        .map(|(entry, row)| PredictionOutput {
            row,
            id: entry.item_id,
        })
        .collect();
    write_batch_outputs(&outputs, args.output_file.as_deref())
}

fn init_logging(level: &str) {
    let filter = EnvFilter::try_new(level).unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .try_init();
}
