use std::env;

use async_channel::Sender;
use axum::{
    body::Body,
    extract::Multipart,
    http::{HeaderMap, StatusCode},
    response::Response,
    routing::post,
    Extension, Router,
};
use clap::Parser;
use nanoid::nanoid;
use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;
use snafu::prelude::*;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::{debug, error};
use tracing_subscriber::EnvFilter;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperError,
};

#[derive(Debug, Clone, Parser)]
#[command(version, about)]
pub struct Args {
    #[arg(short, long, env, default_value = "9091")]
    pub port: u16,
    #[arg(short, long, env, default_value = "/models/ggml-large.bin")]
    pub model: String,
    /// the number of model will be created and work paralleling
    #[arg(short, long, env, default_value_t = 1)]
    pub num_threads: usize,
    /// whether use gpu
    #[arg(short = 'g', long, env, default_value_t = true)]
    pub use_gpu: bool,
    #[arg(long, env, default_value = "请输出简体中文")]
    pub cn_prompt: Option<String>,
}

#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrRequest {
    pub data: Vec<u8>,
    // cn, en, ...
    pub lang: Option<String>,
    pub prompt: Option<String>,
    pub filename: Option<String>,
    pub request_id: Option<String>,
}

#[skip_serializing_none]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AsrResponse {
    pub code: i64,
    pub message: Option<String>,
    pub r#final: Option<AsrRawFinal>,
    pub raw: Option<serde_json::Value>,

    pub time_asr: Option<f64>,
    pub time_pun: Option<f64>,
    #[serde(default)]
    pub mixed_read_time: Vec<(f64, f64)>,
    #[serde(default)]
    pub mixed_ans_time: Vec<(f64, f64)>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AsrRawFinal {
    pub text: String,
    pub sentences: Vec<AsrRawSentence>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AsrRawSentence {
    pub text: String,
    pub start: i64,
    pub end: i64,
    pub words: Option<Vec<AsrRawWord>>,
    #[serde(default)]
    pub is_mixed: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AsrRawWord {
    pub word: String,
    pub start: i64,
    pub end: i64,
}

#[derive(Debug, Snafu)]
pub enum AsrError {
    #[snafu(display("bad request error"))]
    BadRequestError,
    #[snafu(display("read audio error"))]
    ReadAudioError { source: hound::Error },
    #[snafu(display("wav spec error, should match `-c 1 -ar 16000`, got {current_spec}"))]
    WavSpecError { current_spec: String },
    #[snafu(display("create state error"))]
    CreateStateError { source: WhisperError },
    #[snafu(display("predict error"))]
    PredictError { source: WhisperError },
    #[snafu(display("get segment error"))]
    GetSegmentError { source: WhisperError },
    #[snafu(display("send request error"))]
    SendError,
    #[snafu(display("recv response error"))]
    RecvError,
}

impl From<AsrError> for AsrResponse {
    fn from(value: AsrError) -> Self {
        Self {
            code: -2,
            message: Some(value.to_string()),
            ..Default::default()
        }
    }
}

#[derive(Debug)]
pub struct AsrContext {
    pub ctx: WhisperContext,
}

impl AsrContext {
    pub fn new(ctx: WhisperContext) -> Self {
        Self { ctx }
    }
    pub fn predict(&self, req: AsrRequest) -> Result<AsrResponse, AsrError> {
        let AsrRequest {
            data,
            lang,
            filename,
            request_id,
            prompt,
        } = req;
        let request_id = request_id.unwrap_or_else(|| nanoid!(6));
        let lang = lang.unwrap_or_else(|| "cn".to_string());

        let data = std::io::Cursor::new(data);
        let mut reader = hound::WavReader::new(data).context(ReadAudioSnafu)?;

        #[allow(unused_variables)]
        let hound::WavSpec {
            channels,
            sample_rate,
            // bits_per_sample,
            ..
        } = reader.spec();

        if channels != 1 || sample_rate != 16000 {
            let current_spec = format!("c={channels},ar={sample_rate}");
            return Err(AsrError::WavSpecError { current_spec });
        }

        // Convert the audio to floating point samples.
        let audio = whisper_rs::convert_integer_to_float_audio(
            &reader
                .samples::<i16>()
                .map(|s| s.unwrap_or_default())
                // .map(|s| s.expect("invalid sample"))
                .collect::<Vec<_>>(),
        );

        // Convert audio to 16KHz mono f32 samples, as required by the model.
        // These utilities are provided for convenience, but can be replaced with custom conversion logic.
        // SIMD variants of these functions are also available on nightly Rust (see the docs).
        // if channels == 2 {
        //     audio = whisper_rs::convert_stereo_to_mono_audio(&audio)?;
        // } else if channels != 1 {
        //     panic!(">2 channels unsupported");
        // }

        // if sample_rate != 16000 {
        //     panic!("sample rate must be 16KHz");
        // }
        // debug!(
        //     "audio process finished, elapsed:{}s",
        //     tm.elapsed().as_secs_f32()
        // );

        let tm = std::time::Instant::now();
        // Run the model.
        // Create a state
        let mut state = self.ctx.create_state().context(CreateStateSnafu)?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(None);
        if let Some(prompt) = prompt {
            params.set_initial_prompt(&prompt);
        }

        debug!(%request_id, "asr begin");
        state.full(params, &audio[..]).context(PredictSnafu)?;
        let delta = tm.elapsed().as_secs_f32();
        debug!(%request_id, "asr finished, elapsed:{}s", delta);

        // Create a file to write the transcript to.
        // let mut file = File::create("transcript.txt").expect("failed to create file");

        // Iterate through the segments of the transcript.
        let num_segments = state
            .full_n_segments()
            .expect("failed to get number of segments");
        let mut sentences = vec![];
        for i in 0..num_segments {
            // Get the transcribed text and timestamps for the current segment.
            let segment = state.full_get_segment_text(i).context(GetSegmentSnafu)?;
            let segment = segment.trim().to_string();
            let start_timestamp = state.full_get_segment_t0(i).context(GetSegmentSnafu)?;
            let end_timestamp = state.full_get_segment_t1(i).context(GetSegmentSnafu)?;
            let sentence = AsrRawSentence {
                text: segment,
                start: start_timestamp * 10,
                end: end_timestamp * 10,
                ..Default::default()
            };
            sentences.push(sentence);

            // Print the segment to stdout.
            // println!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);

            // Format the segment information as a string.
            // let line = format!("[{} - {}]: {}\n", start_timestamp, end_timestamp, segment);

            // Write the segment information to the file.
            // file.write_all(line.as_bytes())
            //     .expect("failed to write to file");
        }
        let texts: Vec<_> = sentences.iter().map(|it| it.text.clone()).collect();
        let text = texts.join(" ");

        let r#final = AsrRawFinal { text, sentences };
        let response = AsrResponse {
            code: 0,
            r#final: Some(r#final),
            ..Default::default()
        };

        Ok(response)
    }
}

#[derive(Debug, Clone)]
pub struct AsrContextCli {
    pub n: usize,
    pub tx: Sender<(AsrRequest, oneshot::Sender<AsrResponse>)>,
}

impl AsrContextCli {
    pub fn new_from_args(args: Args) -> Self {
        let n = args.num_threads;

        let (tx, rx) = async_channel::bounded::<(AsrRequest, oneshot::Sender<AsrResponse>)>(n + 1);
        let mut ctxs = vec![];
        for _i in 0..n {
            let mut params = WhisperContextParameters::default();
            params.use_gpu = args.use_gpu;
            let ctx = WhisperContext::new_with_params(&args.model, params).unwrap_or_else(|err| {
                panic!(
                    "failed to load model: {}, use_gpu={}, err={}",
                    args.model, args.use_gpu, err
                )
            });
            ctxs.push(ctx);
        }
        for ctx in ctxs {
            let rx = rx.clone();
            tokio::task::spawn_blocking(move || {
                let ctx = AsrContext::new(ctx);
                while let Ok((req, ack)) = rx.recv_blocking() {
                    let response = ctx.predict(req).unwrap_or_else(|err| err.into());
                    if let Err(_err) = ack.send(response) {
                        error!("ack send error");
                    }
                }
            });
        }
        Self { n, tx }
    }

    pub async fn predict(&self, req: AsrRequest) -> Result<AsrResponse, AsrError> {
        let (ack_tx, ack_rx) = oneshot::channel();
        if let Err(_err) = self.tx.send((req, ack_tx)).await {
            return Err(AsrError::SendError);
        }
        let response = match ack_rx.await {
            Ok(it) => it,
            Err(_err) => {
                return Err(AsrError::RecvError);
            }
        };
        Ok(response)
    }
}

pub async fn start_http_server(args: Args) {
    let asr_context_cli = AsrContextCli::new_from_args(args.clone());

    let api_router = Router::new();
    let api_router = api_router.route("/asr", post(asr_handler));
    let app = Router::new();
    let app = app.nest("/api", api_router);
    let app = app
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .layer(Extension(asr_context_cli))
        .layer(Extension(args.clone()));

    let addr = format!("0.0.0.0:{}", args.port);
    let listener = TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|err| panic!("http bind addr failed, addr={addr}, err={err}"));
    axum::serve(listener, app).await.expect("http serve failed");
}

pub async fn asr_handler(
    Extension(asr_context_cli): Extension<AsrContextCli>,
    Extension(args): Extension<Args>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Response {
    let mut f_file_name = None;
    let mut f_file_content = None;
    let mut f_lang = Some("cn".to_string());
    let mut f_prompt = None;
    let field_names = vec!["data", "lang", "prompt"];
    while let Ok(Some(field)) = multipart.next_field().await {
        let field_name = field.name().map(ToOwned::to_owned);
        let file_name = field.file_name().map(ToOwned::to_owned);
        let Some(field_name) = field_name else {
            continue;
        };
        if !field_names.contains(&field_name.as_str()) {
            continue;
        }
        if field_name == "lang" {
            let lang = field.text().await;
            let Ok(lang) = lang else {
                continue;
            };
            f_lang = Some(lang);
            continue;
        }
        if field_name == "prompt" {
            let prompt = field.text().await;
            let Ok(prompt) = prompt else {
                continue;
            };
            f_prompt = Some(prompt);
            continue;
        }
        // 生成唯一文件名
        let r = nanoid!(6);
        let file_name = file_name.unwrap_or("a".to_string());
        let file_name = sanitize_filename::sanitize(&file_name);
        let p = std::path::Path::new(&file_name);
        let file_stem = p
            .file_stem()
            .map(|it| it.to_string_lossy().to_string())
            .unwrap_or("a".to_string());
        let file_ext = p
            .extension()
            .map(|it| it.to_string_lossy().to_string())
            .unwrap_or("docx".to_string());
        let file_name = format!("{file_stem}_{r}.{file_ext}");
        f_file_name = Some(file_name);

        // 读全部文件内容
        let content = match field.bytes().await {
            Ok(it) => it,
            Err(err) => {
                error!(%err, "http read file content error");
                continue;
            }
        };
        let content = content.to_vec();
        f_file_content = Some(content);
    }
    let (Some(f_file_name), Some(f_file_content)) = (f_file_name, f_file_content) else {
        let response = AsrResponse {
            code: 400,
            message: Some(format!("Bad Request")),
            ..Default::default()
        };
        return Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .body(Body::from(serde_json::to_string(&response).unwrap()))
            .unwrap();
    };
    let request_id = headers
        .get("x-request-id")
        .and_then(|it| it.to_str().ok())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| nanoid!(6));

    if f_prompt.is_none() && f_lang == Some("cn".to_string()) {
        f_prompt = args.cn_prompt.clone();
    }

    let req = AsrRequest {
        data: f_file_content,
        lang: f_lang,
        filename: Some(f_file_name),
        request_id: Some(request_id),
        prompt: f_prompt,
    };
    let response = asr_context_cli.predict(req).await;
    let response = response.unwrap_or_else(|err| err.into());
    Response::builder()
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_string(&response).unwrap()))
        .unwrap()
}

#[tokio::main]
async fn main() {
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "debug");
    }
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .try_init();
    let args = Args::parse();
    debug!(?args);
    start_http_server(args).await;
}
