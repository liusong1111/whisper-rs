// This example is not going to build in this folder.
// You need to copy this code into your project and add the dependencies whisper_rs and hound in your cargo.toml

use hound;
use std::fs::File;
use std::io::Write;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Loads a context and model, processes an audio file, and prints the resulting transcript to stdout.
fn main() -> Result<(), &'static str> {
    let model_path = std::env::args().nth(1).expect("model_path");
    let wav_path = std::env::args().nth(2).expect("wav_path");
    let tm = std::time::Instant::now();
    // Load a context and model.
    let mut params = WhisperContextParameters::default();
    params.use_gpu = true;
    let ctx = WhisperContext::new_with_params(
        // "/home/sliu/test/whisper-rs/models/ggml-large.bin",
        &model_path,
        // "/home/sliu/test/whisper-rs/models/ggml-tiny.bin",
        // WhisperContextParameters::default(),
        params,
    )
    .expect("failed to load model");
    // Create a state
    let mut state = ctx.create_state().expect("failed to create key");

    println!("model loaded, elapsed:{}s", tm.elapsed().as_secs_f32());
    // Create a params object for running the model.
    // The number of past samples to consider defaults to 0.
    // let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

    // Edit params as needed.
    // Set the number of threads to use to 1.
    // params.set_n_threads(1);
    // Enable translation.
    // params.set_translate(true);
    // Set the language to translate to to English.
    // params.set_language(Some("en"));
    // Disable anything that prints to stdout.
    // params.set_print_special(false);
    // params.set_print_progress(false);
    // params.set_print_realtime(false);
    // params.set_print_timestamps(false);
    // params.set_tdrz_enable(true);
    // params.set_language(None);
    // params.set_detect_language(true);
    // params.set_suppress_non_speech_tokens(true);
    params.set_initial_prompt("请输出简体中文");

    let tm = std::time::Instant::now();
    // Open the audio file.
    let mut reader = hound::WavReader::open(&wav_path).expect("failed to open file");
    // let mut reader = hound::WavReader::open("1m.wav").expect("failed to open file");
    #[allow(unused_variables)]
    let hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample,
        ..
    } = reader.spec();

    // Convert the audio to floating point samples.
    let mut audio = whisper_rs::convert_integer_to_float_audio(
        &reader
            .samples::<i16>()
            .map(|s| s.expect("invalid sample"))
            .collect::<Vec<_>>(),
    );

    // Convert audio to 16KHz mono f32 samples, as required by the model.
    // These utilities are provided for convenience, but can be replaced with custom conversion logic.
    // SIMD variants of these functions are also available on nightly Rust (see the docs).
    if channels == 2 {
        audio = whisper_rs::convert_stereo_to_mono_audio(&audio)?;
    } else if channels != 1 {
        panic!(">2 channels unsupported");
    }

    if sample_rate != 16000 {
        panic!("sample rate must be 16KHz");
    }
    println!(
        "audio process finished, elapsed:{}s",
        tm.elapsed().as_secs_f32()
    );

    let tm = std::time::Instant::now();
    // Run the model.
    state.full(params, &audio[..]).expect("failed to run model");
    println!("asr finished, elapsed:{}s", tm.elapsed().as_secs_f32());

    // Create a file to write the transcript to.
    let mut file = File::create("transcript.txt").expect("failed to create file");

    println!("start predict");
    // Iterate through the segments of the transcript.
    let num_segments = state
        .full_n_segments()
        .expect("failed to get number of segments");
    for i in 0..num_segments {
        // Get the transcribed text and timestamps for the current segment.
        let segment = state
            .full_get_segment_text(i)
            .expect("failed to get segment");
        let start_timestamp = state
            .full_get_segment_t0(i)
            .expect("failed to get start timestamp");
        let end_timestamp = state
            .full_get_segment_t1(i)
            .expect("failed to get end timestamp");

        // Print the segment to stdout.
        println!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);

        // Format the segment information as a string.
        let line = format!("[{} - {}]: {}\n", start_timestamp, end_timestamp, segment);

        // Write the segment information to the file.
        file.write_all(line.as_bytes())
            .expect("failed to write to file");
    }
    Ok(())
}
