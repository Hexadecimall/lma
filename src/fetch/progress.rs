//! Opt-in CLI rendering; library/desktop downloads never construct this UI.
use std::io::{IsTerminal, Write};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Event {
    Started,
    Bytes(u64),
    Finished(bool),
}

pub(super) fn stderr(filename: &str, total: u64) -> impl FnMut(Event) + use<> {
    let writer = std::io::stderr();
    let terminal = writer.is_terminal();
    let mut progress = Progress::new(writer, terminal, filename, total);
    let start = Instant::now();
    move |event| progress.update(event, start.elapsed())
}

struct Progress<W> {
    writer: Option<W>,
    terminal: bool,
    filename: String,
    total: u64,
    bytes: u64,
    last: Duration,
    started: bool,
    finished: bool,
}

impl<W: Write> Progress<W> {
    fn new(writer: W, terminal: bool, filename: &str, total: u64) -> Self {
        Self {
            writer: Some(writer),
            terminal,
            filename: filename
                .chars()
                .filter(|c| !c.is_control())
                .take(64)
                .collect(),
            total,
            bytes: 0,
            last: Duration::ZERO,
            started: false,
            finished: false,
        }
    }

    fn update(&mut self, event: Event, elapsed: Duration) {
        if self.finished {
            return;
        }
        let state = match event {
            Event::Started => {
                self.started = true;
                None
            }
            Event::Bytes(bytes) => {
                self.bytes = bytes;
                let interval = if self.terminal {
                    Duration::from_millis(150)
                } else {
                    Duration::from_secs(5)
                };
                if !self.started || elapsed.saturating_sub(self.last) < interval {
                    return;
                }
                None
            }
            Event::Finished(success) => {
                self.finished = true;
                if !self.started {
                    return;
                }
                Some(success)
            }
        };
        self.last = elapsed;
        let line = format_progress(&self.filename, self.bytes, self.total, elapsed, state);
        if let Some(writer) = self.writer.as_mut() {
            let result = if self.terminal {
                write!(
                    writer,
                    "\r\x1b[2K{line}{}",
                    if self.finished { "\n" } else { "" }
                )
            } else {
                writeln!(writer, "{line}")
            }
            .and_then(|()| writer.flush());
            // A closed stderr must not interrupt verification or publication.
            if result.is_err() {
                self.writer = None;
            }
        }
    }
}

fn human_bytes(bytes: f64) -> String {
    let mut value = bytes;
    let units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
    let mut unit = 0;
    while value >= 1024.0 && unit < units.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{value:.0} {}", units[unit])
    } else {
        format!("{value:.1} {}", units[unit])
    }
}

// None means transferring/verifying, never successful just because bytes == total.
fn format_progress(
    filename: &str,
    bytes: u64,
    total: u64,
    elapsed: Duration,
    finished: Option<bool>,
) -> String {
    let percent = if finished == Some(true) {
        100
    } else if total == 0 {
        0
    } else {
        (u128::from(bytes) * 100 / u128::from(total)).min(99) as usize
    };
    let filled = percent * 20 / 100;
    let bar = format!("{}{}", "=".repeat(filled), " ".repeat(20 - filled));
    let speed = if elapsed.is_zero() {
        0.0
    } else {
        bytes as f64 / elapsed.as_secs_f64()
    };
    let state = match finished {
        Some(true) => "verified",
        Some(false) => "failed",
        None if bytes == total => "verifying/publishing",
        None => "downloading",
    };
    format!(
        "{filename} [{bar}] {percent:3}% {}/{} ({bytes}/{total} bytes) {}/s {state}",
        human_bytes(bytes as f64),
        human_bytes(total as f64),
        human_bytes(speed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn units_speed_and_percent_are_deterministic_and_overflow_safe() {
        assert_eq!(human_bytes(0.0), "0 B");
        assert_eq!(human_bytes(1024.0), "1.0 KiB");
        assert_eq!(human_bytes(1024.0 * 1024.0 * 1024.0), "1.0 GiB");
        let line = format_progress("model.gguf", 1024, 2048, Duration::from_secs(2), None);
        assert!(line.contains(" 50% 1.0 KiB/2.0 KiB (1024/2048 bytes) 512 B/s downloading"));
        let line = format_progress("m", u64::MAX, u64::MAX, Duration::ZERO, None);
        assert!(line.contains(" 99%"));
        assert!(line.ends_with("0 B/s verifying/publishing"));
        assert!(!line.contains("100%"));
        assert!(format_progress("m", 0, 0, Duration::ZERO, None).contains("  0%"));
    }

    #[test]
    fn tty_redraw_is_throttled_and_finishes_on_its_own_line() {
        let mut progress = Progress::new(Vec::new(), true, "model.gguf", 100);
        progress.update(Event::Started, Duration::ZERO);
        for ms in 1..150 {
            progress.update(Event::Bytes(ms.min(100)), Duration::from_millis(ms));
        }
        assert_eq!(
            progress
                .writer
                .as_ref()
                .unwrap()
                .iter()
                .filter(|b| **b == b'\r')
                .count(),
            1
        );
        progress.update(Event::Bytes(100), Duration::from_millis(150));
        let before = String::from_utf8(progress.writer.as_ref().unwrap().clone()).unwrap();
        assert_eq!(before.matches('\r').count(), 2);
        assert!(before.contains("99%"));
        assert!(!before.contains("100%"));
        progress.update(Event::Finished(true), Duration::from_millis(151));
        progress.update(Event::Finished(true), Duration::from_secs(1));
        let output = String::from_utf8(progress.writer.unwrap()).unwrap();
        assert_eq!(output.matches('\n').count(), 1);
        assert!(output.ends_with("verified\n"));
        assert!(output.contains("100%"));
    }

    #[test]
    fn redirected_output_is_sparse_plain_text_and_errors_end_with_newline() {
        let mut progress = Progress::new(Vec::new(), false, "model.gguf", 100);
        progress.update(Event::Started, Duration::ZERO);
        for ms in 1..=10_000 {
            progress.update(Event::Bytes(100), Duration::from_millis(ms));
        }
        progress.update(Event::Finished(false), Duration::from_millis(10_001));
        let output = String::from_utf8(progress.writer.unwrap()).unwrap();
        assert_eq!(output.lines().count(), 4);
        assert!(!output.contains(['\x1b', '\r']));
        assert!(!output.contains("100%"));
        assert!(!output.contains("verified"));
        assert!(output.ends_with("failed\n"));
    }

    #[test]
    fn no_started_event_means_no_fake_reuse_bar() {
        let mut progress = Progress::new(Vec::new(), true, "model.gguf", 100);
        progress.update(Event::Finished(true), Duration::ZERO);
        assert!(progress.writer.unwrap().is_empty());
    }

    #[test]
    fn broken_writer_does_not_interrupt_download() {
        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("closed stderr"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                panic!("failed write must not be flushed")
            }
        }
        let mut progress = Progress::new(Broken, true, "m", 1);
        progress.update(Event::Started, Duration::ZERO);
        assert!(progress.writer.is_none());
        progress.update(Event::Bytes(1), Duration::from_secs(1));
        progress.update(Event::Finished(true), Duration::from_secs(1));
    }
}
