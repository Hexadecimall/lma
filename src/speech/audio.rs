use super::{MAX_AUDIO_SECONDS, SAMPLE_RATE};
use hound::{SampleFormat, WavReader};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::{
    io::{BufReader, Read},
    path::Path,
};

const MAX_WAV_BYTES: u64 = 512 * 1024 * 1024;
const BLOCK: usize = 4096;

pub(super) fn read_wav(path: &Path) -> Result<Vec<f32>, String> {
    let file = crate::local_file::open_regular(path)?;
    let metadata = file
        .metadata()
        .map_err(|e| format!("cannot inspect WAV: {e}"))?;
    if !metadata.is_file() || metadata.len() > MAX_WAV_BYTES {
        return Err("audio must be a regular WAV file no larger than 512 MiB".into());
    }
    // Bound reads as well as metadata, including skipped RIFF chunks and growing files.
    let mut reader = WavReader::new(BufReader::new(file).take(MAX_WAV_BYTES)).map_err(|e| {
        format!(
            "invalid WAV {}: {e}; expected PCM16 or float32 WAV",
            path.display()
        )
    })?;
    let spec = reader.spec();
    if !matches!(spec.channels, 1 | 2) {
        return Err("WAV must contain one or two channels".into());
    }
    if !(8_000..=192_000).contains(&spec.sample_rate) {
        return Err("WAV sample rate must be between 8000 and 192000 Hz".into());
    }
    if !reader.len().is_multiple_of(u32::from(spec.channels)) {
        return Err("WAV data ends in an incomplete audio frame".into());
    }
    let frames = reader.duration() as usize;
    if frames == 0 || frames > spec.sample_rate as usize * MAX_AUDIO_SECONDS {
        return Err("WAV must contain audio and be no longer than 30 minutes".into());
    }
    let result = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => decode(
            reader
                .samples::<i16>()
                .map(|v| v.map(|v| f32::from(v) / 32768.0).map_err(|e| e.to_string())),
            frames,
            spec.channels as usize,
            spec.sample_rate,
        ),
        (SampleFormat::Float, 32) => decode(
            reader
                .samples::<f32>()
                .map(|v| v.map_err(|e| e.to_string())),
            frames,
            spec.channels as usize,
            spec.sample_rate,
        ),
        _ => Err("unsupported WAV encoding; use uncompressed PCM16 or IEEE float32".into()),
    };
    result.map_err(|e| format!("cannot decode WAV {}: {e}", path.display()))
}

fn decode(
    mut samples: impl Iterator<Item = Result<f32, String>>,
    frames: usize,
    channels: usize,
    rate: u32,
) -> Result<Vec<f32>, String> {
    let mono = (0..frames).map(|_| {
        let mut sum = 0.0;
        for _ in 0..channels {
            let value = samples.next().ok_or("truncated WAV samples")??;
            if !value.is_finite() || value.abs() > 1.0 {
                return Err("float WAV samples must be finite and in [-1, 1]".into());
            }
            sum += value;
        }
        Ok(sum / channels as f32)
    });
    resample(mono, frames, rate)
}

fn resample(
    mut input: impl Iterator<Item = Result<f32, String>>,
    frames: usize,
    rate: u32,
) -> Result<Vec<f32>, String> {
    if rate as usize == SAMPLE_RATE {
        return input.collect();
    }
    let target = (frames as u64 * SAMPLE_RATE as u64).div_ceil(u64::from(rate)) as usize;
    let parameters = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Cubic,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut converter = SincFixedIn::<f32>::new(
        SAMPLE_RATE as f64 / f64::from(rate),
        1.0,
        parameters,
        BLOCK,
        1,
    )
    .map_err(|e| format!("cannot initialize audio resampler: {e}"))?;
    let mut skip = converter.output_delay();
    let mut output = Vec::with_capacity(target);
    let mut buffer = vec![0.0; BLOCK];
    let blocks = frames.div_ceil(BLOCK) + (256_usize.div_ceil(BLOCK)) + 2;
    for _ in 0..blocks {
        buffer.fill(0.0);
        for value in &mut buffer {
            match input.next() {
                Some(sample) => *value = sample?,
                None => break,
            }
        }
        let converted = converter
            .process(&[buffer.as_slice()], None)
            .map_err(|e| format!("audio resampling failed: {e}"))?;
        let channel = &converted[0];
        let discard = skip.min(channel.len());
        skip -= discard;
        // Sinc filters can overshoot full-scale samples; keep the public float contract.
        output.extend(
            channel[discard..]
                .iter()
                .take(target - output.len())
                .map(|v| v.clamp(-1.0, 1.0)),
        );
        if output.len() == target {
            return Ok(output);
        }
    }
    Err("audio resampler could not flush its final samples".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::{WavSpec, WavWriter};
    use std::fs::File;

    fn wav(
        channels: u16,
        format: SampleFormat,
        bits: u16,
        rate: u32,
    ) -> (
        tempfile::TempDir,
        std::path::PathBuf,
        WavWriter<std::io::BufWriter<File>>,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.wav");
        let writer = WavWriter::create(
            &path,
            WavSpec {
                channels,
                sample_rate: rate,
                bits_per_sample: bits,
                sample_format: format,
            },
        )
        .unwrap();
        (dir, path, writer)
    }

    #[test]
    fn reads_pcm16_and_downmixes_stereo() {
        let (_dir, path, mut writer) = wav(2, SampleFormat::Int, 16, 16000);
        for value in [32767_i16, -32768, 16384, 16384] {
            writer.write_sample(value).unwrap();
        }
        writer.finalize().unwrap();
        let audio = read_wav(&path).unwrap();
        assert_eq!(audio.len(), 2);
        assert!(audio[0].abs() < 0.0001);
        assert_eq!(audio[1], 0.5);
    }

    #[test]
    fn float_wav_and_invalid_float() {
        let (_dir, path, mut writer) = wav(1, SampleFormat::Float, 32, 16000);
        writer.write_sample(0.25_f32).unwrap();
        writer.finalize().unwrap();
        assert_eq!(read_wav(&path).unwrap(), vec![0.25]);
        let (_dir, path, mut writer) = wav(1, SampleFormat::Float, 32, 16000);
        writer.write_sample(f32::NAN).unwrap();
        writer.finalize().unwrap();
        assert!(read_wav(&path).unwrap_err().contains("finite"));
    }

    #[test]
    fn rejects_unsupported_empty_and_truncated_audio() {
        let (_dir, path, mut writer) = wav(1, SampleFormat::Int, 24, 16000);
        writer.write_sample(1_i32).unwrap();
        writer.finalize().unwrap();
        assert!(read_wav(&path).unwrap_err().contains("PCM16"));
        let (_dir, path, writer) = wav(1, SampleFormat::Int, 16, 16000);
        writer.finalize().unwrap();
        assert!(read_wav(&path).is_err());
        assert!(decode(std::iter::empty(), 10, 1, 16000).is_err());
        let (_dir, path, mut writer) = wav(1, SampleFormat::Int, 16, 16000);
        writer.write_sample(1_i16).unwrap();
        writer.finalize().unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(44)
            .unwrap();
        assert!(read_wav(&path).is_err());
    }

    fn tone(rate: u32, hz: f32) -> Vec<f32> {
        (0..rate)
            .map(|i| 0.5 * (std::f32::consts::TAU * hz * i as f32 / rate as f32).sin())
            .collect()
    }

    #[test]
    fn resampling_preserves_duration_and_suppresses_aliasing() {
        let low = tone(48000, 1000.0);
        let high = tone(48000, 12000.0);
        let low = resample(low.into_iter().map(Ok), 48000, 48000).unwrap();
        let high = resample(high.into_iter().map(Ok), 48000, 48000).unwrap();
        assert_eq!(low.len(), 16000);
        assert_eq!(high.len(), 16000);
        let rms = |v: &[f32]| {
            (v[512..v.len() - 512].iter().map(|s| s * s).sum::<f32>() / (v.len() - 1024) as f32)
                .sqrt()
        };
        assert!(rms(&low) > 0.3);
        assert!(rms(&high) < 0.005);
        for rate in [8000, 22050, 44100, 192000] {
            let data = tone(rate, 1000.0);
            assert_eq!(
                resample(data.into_iter().map(Ok), rate as usize, rate)
                    .unwrap()
                    .len(),
                16000
            );
        }
        assert_eq!(resample([Ok(0.5)].into_iter(), 1, 48000).unwrap().len(), 1);
    }

    #[test]
    fn input_bytes_are_bounded_before_decode() {
        let file = tempfile::NamedTempFile::new().unwrap();
        file.as_file().set_len(MAX_WAV_BYTES + 1).unwrap();
        assert!(read_wav(file.path()).unwrap_err().contains("512 MiB"));
    }
}
