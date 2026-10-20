use parakeet_rs::{ExecutionConfig, ParakeetEOU};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = env::args().collect();
    if a.len() != 5 {
        eprintln!("usage: eou_dump <model_dir> <wav_path> <chunk_samples> <reset_on_eou 0|1>");
        std::process::exit(2);
    }
    let chunk_n: usize = a[3].parse()?;
    let reset = a[4] == "1";
    let mut reader = hound::WavReader::open(&a[2])?;
    let spec = reader.spec();
    let ch = spec.channels as usize;
    let all: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let d = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / d))
                .collect::<Result<_, _>>()?
        }
    };
    let audio: Vec<f32> = all.chunks(ch).map(|f| f[0]).collect();
    if spec.sample_rate != 16000 {
        eprintln!("warning: sample rate {} != 16000", spec.sample_rate);
    }
    let mut m = ParakeetEOU::from_pretrained(&a[1], Some(ExecutionConfig::new().with_environment(parakeet_rs::ort::init().build()?)))?;
    let mut i = 0usize;
    let mut pos = 0usize;
    while pos < audio.len() {
        let end = (pos + chunk_n).min(audio.len());
        let mut c = audio[pos..end].to_vec();
        c.resize(chunk_n, 0.0);
        let t = m.transcribe(&c, reset)?;
        println!(
            "{}",
            serde_json::json!({"call": i, "end_sample": pos + chunk_n, "text": t})
        );
        i += 1;
        pos += chunk_n;
    }
    println!("{}", serde_json::json!({"done": true}));
    Ok(())
}
