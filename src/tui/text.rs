//! Terminal-safe text and cell-aware wrapping. Model output is never interpreted.
use ratatui::{style::Style, text::Span};
use unicode_width::UnicodeWidthStr;

pub fn sanitize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\u{1b}' => match chars.next() {
                Some('[') => {
                    for c in chars.by_ref() {
                        if ('\u{40}'..='\u{7e}').contains(&c) {
                            break;
                        }
                    }
                }
                Some(']') | Some('P') | Some('X') | Some('^') | Some('_') => {
                    while let Some(c) = chars.next() {
                        if c == '\u{7}' || c == '\u{9c}' {
                            break;
                        }
                        if c == '\u{1b}' && chars.peek() == Some(&'\\') {
                            chars.next();
                            break;
                        }
                    }
                }
                Some(c) if (' '..='/').contains(&c) => {
                    for c in chars.by_ref() {
                        if ('0'..='~').contains(&c) {
                            break;
                        }
                    }
                }
                _ => {}
            },
            '\u{9b}' => {
                for c in chars.by_ref() {
                    if ('\u{40}'..='\u{7e}').contains(&c) {
                        break;
                    }
                }
            }
            '\u{9d}' | '\u{90}' | '\u{98}' | '\u{9e}' | '\u{9f}' => {
                while let Some(c) = chars.next() {
                    if c == '\u{7}' || c == '\u{9c}' {
                        break;
                    }
                    if c == '\u{1b}' && chars.peek() == Some(&'\\') {
                        chars.next();
                        break;
                    }
                }
            }
            '\n' => out.push('\n'),
            '\t' => out.push_str("    "),
            c if c.is_control()
                || matches!(c, '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}') => {}
            c => out.push(c),
        }
    }
    out
}

pub fn one_line(text: &str) -> String {
    sanitize(text).replace('\n', " ")
}

/// Returns wrapped lines and a cell coordinate for a byte-index cursor.
pub fn wrap(text: &str, width: u16, cursor: usize) -> (Vec<String>, (usize, u16)) {
    let width = usize::from(width.max(1));
    let mut lines = vec![String::new()];
    let mut col = 0;
    let mut position = (0, 0);
    let mut index = 0;
    for (line_index, logical_line) in text.split('\n').enumerate() {
        if line_index > 0 {
            if index == cursor {
                position = (lines.len() - 1, col as u16);
            }
            lines.push(String::new());
            col = 0;
            index += 1;
        }
        let span = Span::raw(logical_line);
        for grapheme in span.styled_graphemes(Style::default()) {
            let cells = grapheme.symbol.width().min(width);
            if col + cells > width {
                lines.push(String::new());
                col = 0;
            }
            if (index..index + grapheme.symbol.len()).contains(&cursor) {
                position = (lines.len() - 1, col as u16);
            }
            lines.last_mut().unwrap().push_str(grapheme.symbol);
            col += cells;
            index += grapheme.symbol.len();
        }
    }
    if cursor == text.len() {
        if col >= width {
            lines.push(String::new());
            col = 0;
        }
        position = (lines.len() - 1, col as u16);
    }
    (lines, position)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strips_terminal_protocols_and_controls() {
        assert_eq!(
            sanitize("a\x1b]52;c;YQ==\x07b\x1b[31m赤\x1b[0m\0\r\n\t"),
            "ab赤\n    "
        );
        assert_eq!(
            sanitize("a\x1b]0;title\x1b\\b\u{9b}31mc\u{9d}x\u{9c}d"),
            "abcd"
        );
        assert_eq!(sanitize("x\x1bPpayload\x1b\\y\u{202e}z"), "xyz");
        assert_eq!(sanitize("safe\x1b]unterminated"), "safe");
    }
    #[test]
    fn wraps_by_cells_not_bytes() {
        let (lines, cursor) = wrap("a界🙂z", 4, "a界🙂".len());
        assert_eq!(lines, ["a界", "🙂z"]);
        assert_eq!(cursor, (1, 2));
        assert_eq!(wrap("界界", 4, 6).1, (1, 0));
        assert_eq!(wrap("👩‍💻e\u{301}z", 4, "👩‍💻e\u{301}".len()).1, (0, 3));
        assert_eq!(wrap("a\nb", 4, 2).1, (1, 0));
    }
}
