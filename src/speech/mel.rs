//! OpenAI Whisper frontend: Slaney-normalized mel filters and centered, periodic
//! Hann-window STFT. Construct filters locally, without a downloaded auxiliary file.
use candle_transformers::models::whisper::{HOP_LENGTH, N_FFT, N_FRAMES, N_SAMPLES};
use rustfft::{FftPlanner, num_complex::Complex32};

const BINS: usize = N_FFT / 2 + 1;

pub(super) fn filters(mels: usize) -> Vec<f32> {
    fn hz_to_mel(hz: f64) -> f64 {
        if hz < 1000.0 {
            hz / (200.0 / 3.0)
        } else {
            15.0 + (hz / 1000.0).ln() / (6.4_f64.ln() / 27.0)
        }
    }
    fn mel_to_hz(mel: f64) -> f64 {
        if mel < 15.0 {
            mel * (200.0 / 3.0)
        } else {
            1000.0 * ((mel - 15.0) * (6.4_f64.ln() / 27.0)).exp()
        }
    }
    let max = hz_to_mel(8000.0);
    let edges: Vec<f64> = (0..mels + 2)
        .map(|i| mel_to_hz(max * i as f64 / (mels + 1) as f64))
        .collect();
    let mut weights = vec![0.0; mels * BINS];
    for mel in 0..mels {
        let norm = 2.0 / (edges[mel + 2] - edges[mel]);
        for bin in 0..BINS {
            let hz = bin as f64 * 16000.0 / N_FFT as f64;
            let lower = (hz - edges[mel]) / (edges[mel + 1] - edges[mel]);
            let upper = (edges[mel + 2] - hz) / (edges[mel + 2] - edges[mel + 1]);
            // librosa stores each triangle in f32 before applying its f64 normalization.
            let triangle = lower.min(upper).max(0.0) as f32;
            weights[mel * BINS + bin] = (f64::from(triangle) * norm) as f32;
        }
    }
    weights
}

pub(super) fn spectrogram(
    samples: &[f32],
    filters: &[f32],
    mels: usize,
    threads: usize,
) -> Vec<f32> {
    let fft = FftPlanner::<f32>::new().plan_fft_forward(N_FFT);
    let hann: Vec<f32> = (0..N_FFT)
        .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / N_FFT as f32).cos())
        .collect();
    let workers = threads.clamp(1, N_FRAMES);
    let per_worker = N_FRAMES.div_ceil(workers);
    let pieces = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for start in (0..N_FRAMES).step_by(per_worker) {
            let end = (start + per_worker).min(N_FRAMES);
            let fft = &fft;
            let hann = &hann;
            handles.push(scope.spawn(move || {
                let mut buffer = vec![Complex32::default(); N_FFT];
                let mut scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
                let mut power = [0.0; BINS];
                let mut output = Vec::with_capacity((end - start) * mels);
                for frame in start..end {
                    for (i, value) in buffer.iter_mut().enumerate() {
                        let position = (frame * HOP_LENGTH + i) as isize - (N_FFT / 2) as isize;
                        // Reflect-pad the 30-second zero-padded waveform, as torch.stft does.
                        let position = if position < 0 {
                            -position
                        } else if position >= N_SAMPLES as isize {
                            2 * N_SAMPLES as isize - 2 - position
                        } else {
                            position
                        } as usize;
                        *value = Complex32::new(
                            samples.get(position).copied().unwrap_or(0.0) * hann[i],
                            0.0,
                        );
                    }
                    fft.process_with_scratch(&mut buffer, &mut scratch);
                    for (power, value) in power.iter_mut().zip(&buffer[..BINS]) {
                        *power = value.norm_sqr();
                    }
                    for filter in filters.as_chunks::<BINS>().0 {
                        let energy: f32 = filter.iter().zip(&power).map(|(w, p)| w * p).sum();
                        output.push(energy.max(1e-10).log10());
                    }
                }
                (start, output)
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("speech spectral worker panicked"))
            .collect::<Vec<_>>()
    });
    let max = pieces
        .iter()
        .flat_map(|(_, values)| values.iter())
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut output = vec![0.0; mels * N_FRAMES];
    for (start, values) in pieces {
        for (offset, frame) in values.chunks_exact(mels).enumerate() {
            for (mel, value) in frame.iter().enumerate() {
                output[mel * N_FRAMES + start + offset] = (value.max(max - 8.0) + 4.0) / 4.0;
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slaney_filters_match_known_whisper_coefficients() {
        let filter = filters(80);
        assert_eq!(filter.len(), 80 * 201);
        assert_eq!(filter[0], 0.0);
        assert!((filter[1] - 0.024862594).abs() < 1e-7);
        assert!(filter.iter().all(|w| w.is_finite() && *w >= 0.0));
        assert_eq!(filter[202], 0.001_990_822);
        assert_eq!(filter[203], 0.022871772);
        assert_eq!(filter[16078], 0.000448759);
        let filter = filters(128);
        assert_eq!(filter.len(), 128 * 201);
        assert_eq!(filter[1], 0.012373987);
        assert_eq!(filter[202], 0.030392565);
        assert_eq!(filter[25726], 0.0011142802);
    }

    #[test]
    fn matches_torch_centered_stft_fixtures() {
        // Reference: torch.stft(n_fft=400, hop_length=160, center=True,
        // pad_mode="reflect", window=torch.hann_window(400)), abs()**2,
        // drop the last frame, then Slaney mel filters and Whisper log scaling.
        // 80 bins exercises a short waveform padded to 30 s; 128 exercises the
        // right reflection of an exactly 30 s waveform. No external assets needed.
        for (mels, length, expected) in [
            (
                80,
                800,
                [
                    -1.4017105,
                    0.1273427,
                    0.5982895,
                    0.4748127,
                    0.49380416,
                    -0.11348224,
                    -0.13475573,
                    -1.4017105,
                    -1.4017105,
                    -1.4017105,
                    -1.4017105,
                    -1.4017105,
                ],
            ),
            (
                128,
                N_SAMPLES,
                [
                    -1.4011514,
                    0.1165154,
                    0.5988486,
                    0.39905345,
                    0.48732048,
                    -0.11343563,
                    -0.2105149,
                    -1.4011514,
                    -0.088950515,
                    0.43477535,
                    0.48810685,
                    0.49789518,
                ],
            ),
        ] {
            let mut samples = vec![0.0; length];
            for (index, value) in [
                (0, 0.5),
                (1, -0.25),
                (199, 0.125),
                (200, -0.75),
                (799, 0.375),
            ] {
                samples[index] = value;
            }
            if length == N_SAMPLES {
                samples[479840] = 0.625;
                samples[479999] = -0.5;
            }
            let actual = spectrogram(&samples, &filters(mels), mels, 2);
            assert_eq!(actual.len(), mels * N_FRAMES);
            let positions = [
                (0, 0),
                (mels / 2, 0),
                (mels - 1, 0),
                (0, 1),
                (mels / 2, 1),
                (mels - 1, 4),
                (0, 6),
                (mels / 2, 100),
                (0, 2998),
                (0, 2999),
                (mels / 2, 2999),
                (mels - 1, 2999),
            ];
            for ((mel, frame), expected) in positions.into_iter().zip(expected) {
                let actual = actual[mel * N_FRAMES + frame];
                assert!(
                    (actual - expected).abs() < 2e-5,
                    "{mels} bins, mel {mel}, frame {frame}: {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn silence_has_fixed_shape_and_normalization() {
        let filter = filters(80);
        let mel = spectrogram(&[0.0; 160], &filter, 80, 2);
        assert_eq!(mel.len(), 80 * 3000);
        assert!(mel.iter().all(|v| (*v + 1.5).abs() < 1e-6));
    }

    #[test]
    fn frontend_is_independent_of_worker_count() {
        let filter = filters(80);
        let samples: Vec<f32> = (0..800).map(|i| (i as f32 * 0.3).sin() * 0.5).collect();
        assert_eq!(
            spectrogram(&samples, &filter, 80, 1),
            spectrogram(&samples, &filter, 80, 3)
        );
    }
}
