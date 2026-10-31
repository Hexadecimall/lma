use crossterm::{
    cursor::Show,
    event::{
        DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
        KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
    },
    execute,
    terminal::{self, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{Frame, Terminal, backend::CrosstermBackend, style::Color};
use std::{
    io::{self, IsTerminal, Stdout},
    sync::{
        Once, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

static ACTIVE: AtomicBool = AtomicBool::new(false);
static PANIC_HOOK: Once = Once::new();
static ENHANCED_KEYBOARD: OnceLock<bool> = OnceLock::new();
static KEYBOARD_ACTIVE: AtomicBool = AtomicBool::new(false);

pub fn require_tty(input: bool, output: bool) -> Result<(), String> {
    if input && output {
        Ok(())
    } else {
        Err("Interactive launch needs a terminal on stdin and stdout. Run `lma s MODEL` for non-interactive hosting, or open a terminal for chat/setup.".into())
    }
}

pub fn check_tty() -> Result<(), String> {
    require_tty(io::stdin().is_terminal(), io::stdout().is_terminal())
}

fn restore() {
    if KEYBOARD_ACTIVE.swap(false, Ordering::SeqCst) {
        let _ = execute!(io::stdout(), PopKeyboardEnhancementFlags);
    }
    let _ = terminal::disable_raw_mode();
    let _ = execute!(
        io::stdout(),
        DisableBracketedPaste,
        DisableMouseCapture,
        Show,
        LeaveAlternateScreen
    );
}

struct Restore;
impl Drop for Restore {
    fn drop(&mut self) {
        if ACTIVE.swap(false, Ordering::SeqCst) {
            restore();
        }
    }
}

pub struct Screen {
    pub terminal: Terminal<CrosstermBackend<Stdout>>,
    _restore: Restore,
    color: bool,
}

fn monochrome(frame: &mut Frame) {
    for cell in &mut frame.buffer_mut().content {
        cell.set_fg(Color::Reset).set_bg(Color::Reset);
    }
}

impl Screen {
    pub fn draw(&mut self, render: impl FnOnce(&mut Frame)) -> io::Result<()> {
        self.terminal.draw(|frame| {
            render(frame);
            if !self.color {
                monochrome(frame);
            }
        })?;
        Ok(())
    }
    pub fn enter() -> Result<Self, String> {
        PANIC_HOOK.call_once(|| {
            let previous = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                if ACTIVE.swap(false, Ordering::SeqCst) {
                    restore();
                }
                previous(info);
            }));
        });
        if ACTIVE
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err("another terminal workspace is already active".into());
        }
        // Construct the guard before any fallible terminal operation.
        let guard = Restore;
        terminal::enable_raw_mode().map_err(|e| format!("cannot enable terminal input: {e}"))?;
        let mut stdout = io::stdout();
        execute!(
            stdout,
            EnterAlternateScreen,
            EnableBracketedPaste,
            EnableMouseCapture
        )
        .map_err(|e| format!("cannot enter terminal workspace: {e}"))?;
        // Legacy terminals collapse Shift+Enter into Enter; query once and retain
        // Alt+Enter as a portable fallback rather than changing ordinary Enter.
        if *ENHANCED_KEYBOARD
            .get_or_init(|| terminal::supports_keyboard_enhancement().unwrap_or(false))
        {
            KEYBOARD_ACTIVE.store(true, Ordering::SeqCst);
            execute!(
                stdout,
                PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
            )
            .map_err(|e| format!("cannot enable enhanced keyboard input: {e}"))?;
        }
        let mut terminal =
            Terminal::new(CrosstermBackend::new(stdout)).map_err(|e| e.to_string())?;
        terminal.hide_cursor().map_err(|e| e.to_string())?;
        Ok(Self {
            terminal,
            _restore: guard,
            color: crate::terminal_style::TerminalStyle::stdout().color_enabled(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn monochrome_keeps_formatting_but_removes_foreground_and_background_colors() {
        use ratatui::{
            backend::TestBackend,
            style::{Modifier, Style},
            widgets::Paragraph,
        };
        let mut terminal = Terminal::new(TestBackend::new(20, 3)).unwrap();
        terminal
            .draw(|frame| {
                frame.render_widget(
                    Paragraph::new("bold").style(
                        Style::default()
                            .fg(Color::Red)
                            .bg(Color::Blue)
                            .add_modifier(Modifier::BOLD),
                    ),
                    frame.area(),
                );
                monochrome(frame);
            })
            .unwrap();
        let cell = &terminal.backend().buffer()[(0, 0)];
        assert_eq!(cell.fg, Color::Reset);
        assert_eq!(cell.bg, Color::Reset);
        assert!(cell.modifier.contains(Modifier::BOLD));
    }
    #[test]
    fn non_tty_is_rejected_without_touching_terminal() {
        for (input, output) in [(false, false), (false, true), (true, false)] {
            let error = require_tty(input, output).unwrap_err();
            assert!(error.contains("terminal"));
            assert!(error.contains("`lma s MODEL`"));
            assert!(!error.contains("--mode"));
            assert!(!ACTIVE.load(Ordering::SeqCst));
        }
        assert!(require_tty(true, true).is_ok());
    }
}
