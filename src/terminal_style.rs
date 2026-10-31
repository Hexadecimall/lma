//! Human-facing CLI styling. Machine output must bypass this module.
use std::{
    env,
    ffi::OsStr,
    io::{self, IsTerminal},
};

use crossterm::style::{Attribute, Color, ResetColor, SetAttribute, SetForegroundColor};

#[derive(Clone, Copy, Debug)]
pub struct TerminalStyle {
    terminal: bool,
    color: bool,
}

impl TerminalStyle {
    pub fn stdout() -> Self {
        Self::detect(io::stdout().is_terminal())
    }

    pub fn stderr() -> Self {
        Self::detect(io::stderr().is_terminal())
    }

    fn detect(terminal: bool) -> Self {
        Self::for_environment(
            terminal,
            env::var_os("NO_COLOR").as_deref(),
            env::var_os("TERM").as_deref(),
        )
    }

    pub(crate) fn for_environment(
        terminal: bool,
        no_color: Option<&OsStr>,
        term: Option<&OsStr>,
    ) -> Self {
        Self {
            terminal,
            color: terminal
                && !no_color.is_some_and(|value| !value.is_empty())
                && term != Some(OsStr::new("dumb")),
        }
    }

    pub fn is_terminal(self) -> bool {
        self.terminal
    }

    pub fn color_enabled(self) -> bool {
        self.color
    }

    fn paint(self, text: &str, color: Color, bold: bool) -> String {
        if !self.color || text.is_empty() {
            return text.into();
        }
        let foreground = SetForegroundColor(color);
        if bold {
            format!(
                "{foreground}{}{text}{}{ResetColor}",
                SetAttribute(Attribute::Bold),
                SetAttribute(Attribute::NormalIntensity),
            )
        } else {
            format!("{foreground}{text}{ResetColor}")
        }
    }

    /// Callers must sanitize untrusted text before applying styles.
    pub fn heading(self, text: &str) -> String {
        self.paint(text, Color::Cyan, true)
    }

    pub fn accent(self, text: &str) -> String {
        self.paint(text, Color::Cyan, false)
    }

    pub fn muted(self, text: &str) -> String {
        self.paint(text, Color::DarkGrey, false)
    }

    pub fn error(self, message: &str) -> String {
        format!("{} {}", self.paint("lma:", Color::Red, true), safe(message))
    }

    pub fn warning(self, message: &str) -> String {
        format!(
            "{} {}",
            self.paint("lma:", Color::Yellow, true),
            safe(message)
        )
    }

    pub fn version(self, version: &str) -> String {
        format!("{} {}", self.heading("lma"), safe(version))
    }

    /// Style trusted, static help without dropping or reflowing its detailed text.
    pub fn help(self, text: &str) -> String {
        let mut output = String::with_capacity(text.len());
        for (index, line) in text.split_inclusive('\n').enumerate() {
            let (line, newline) = line
                .strip_suffix('\n')
                .map_or((line, ""), |line| (line, "\n"));
            let trimmed = line.trim_start();
            if index == 0
                || (!line.starts_with(' ')
                    && line.chars().any(|c| c.is_ascii_uppercase())
                    && !line.chars().any(|c| c.is_ascii_lowercase()))
            {
                output.push_str(&self.heading(line));
            } else if trimmed.starts_with('-') || trimmed.starts_with("lma ") {
                output.push_str(&line[..line.len() - trimmed.len()]);
                let end = trimmed.find("  ").unwrap_or(trimmed.len());
                output.push_str(&self.accent(&trimmed[..end]));
                output.push_str(&trimmed[end..]);
            } else if let Some((tree, command)) = line.split_once("─ ") {
                output.push_str(&self.muted(&format!("{tree}─ ")));
                let end = command.find("  ").unwrap_or(command.len());
                output.push_str(&self.accent(&command[..end]));
                output.push_str(&command[end..]);
            } else {
                output.push_str(line);
            }
            output.push_str(newline);
        }
        output
    }
}

/// Escape control and directional-formatting characters in names, paths and errors.
/// Keep these visible even in pipes; JSON uses serde's own lossless escaping instead.
pub fn safe(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    for c in text.chars() {
        if c.is_control()
            || matches!(c, '\u{061c}' | '\u{200e}'..='\u{200f}' | '\u{2028}'..='\u{202e}' | '\u{2066}'..='\u{2069}')
        {
            output.extend(c.escape_default());
        } else {
            output.push(c);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_requires_a_terminal_and_respects_nonempty_no_color_and_dumb() {
        for (terminal, no_color, term, colored) in [
            (true, None, None, true),
            (true, Some(""), Some("xterm-256color"), true),
            (true, Some("0"), Some("xterm-256color"), false),
            (true, Some("1"), None, false),
            (true, None, Some("dumb"), false),
            (false, None, Some("xterm-256color"), false),
            (false, Some(""), None, false),
        ] {
            let style = TerminalStyle::for_environment(
                terminal,
                no_color.map(OsStr::new),
                term.map(OsStr::new),
            );
            assert_eq!(style.heading("USAGE").contains('\x1b'), colored);
            assert_eq!(style.is_terminal(), terminal);
        }
    }

    #[test]
    fn untrusted_diagnostics_cannot_inject_terminal_controls() {
        let style = TerminalStyle::for_environment(false, None, None);
        let message = "bad\x1b]8;;https://example.org\x07name\r\n\t\u{009b}\u{202e}\u{2066}";
        let error = style.error(message);
        assert!(error.starts_with("lma: bad\\u{1b}"));
        assert!(!error.chars().any(char::is_control));
        assert!(!error.contains('\u{202e}'));
        assert!(!error.contains('\u{2066}'));
    }

    #[test]
    fn plain_help_and_version_are_unchanged() {
        let style = TerminalStyle::for_environment(false, None, None);
        let help = "lma 0.2.0\n\nUSAGE\n  lma l\n  -j  JSON\n├─ l [FILTER]  Models\n";
        assert_eq!(style.help(help), help);
        assert_eq!(style.version("0.2.0"), "lma 0.2.0");
    }
}
