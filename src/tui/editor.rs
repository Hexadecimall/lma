use super::text::sanitize;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{style::Style, text::Span};
use unicode_width::UnicodeWidthStr;

pub const INPUT_LIMIT: usize = 256 * 1024;

#[derive(Default)]
pub struct Editor {
    pub text: String,
    pub cursor: usize,
}

impl Editor {
    pub fn new(text: String) -> Self {
        let text = sanitize(&text);
        let cursor = text.len();
        Self { text, cursor }
    }
    pub fn insert(&mut self, text: &str) {
        let text = sanitize(text);
        if self.text.len().saturating_add(text.len()) <= INPUT_LIMIT {
            self.text.insert_str(self.cursor, &text);
            self.cursor += text.len();
            self.snap_cursor();
        }
    }
    pub fn key(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('a') if ctrl => self.cursor = self.line_start(),
            KeyCode::Char('e') if ctrl => self.cursor = self.line_end(),
            KeyCode::Char('u') if ctrl => {
                let start = self.line_start();
                self.text.replace_range(start..self.cursor, "");
                self.cursor = start;
            }
            KeyCode::Char(c)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                self.insert(&c.to_string())
            }
            KeyCode::Left => self.cursor = self.previous(),
            KeyCode::Right => self.cursor = self.next(),
            KeyCode::Home => self.cursor = self.line_start(),
            KeyCode::End => self.cursor = self.line_end(),
            KeyCode::Backspace if self.cursor > 0 => {
                let start = self.previous();
                self.text.replace_range(start..self.cursor, "");
                self.cursor = start;
            }
            KeyCode::Delete if self.cursor < self.text.len() => {
                let end = self.next();
                self.text.replace_range(self.cursor..end, "");
            }
            KeyCode::Up => self.vertical(false),
            KeyCode::Down => self.vertical(true),
            _ => {}
        }
        self.snap_cursor();
    }
    fn previous(&self) -> usize {
        if self.cursor == self.line_start() {
            return self.cursor.saturating_sub(1);
        }
        let span = Span::raw(&self.text[self.line_start()..self.cursor]);
        self.cursor
            - span
                .styled_graphemes(Style::default())
                .last()
                .map_or(0, |g| g.symbol.len())
    }
    fn next(&self) -> usize {
        if self.cursor == self.line_end() {
            return (self.cursor + 1).min(self.text.len());
        }
        let span = Span::raw(&self.text[self.cursor..self.line_end()]);
        self.cursor
            + span
                .styled_graphemes(Style::default())
                .next()
                .map_or(0, |g| g.symbol.len())
    }
    fn snap_cursor(&mut self) {
        let start = self.line_start();
        let span = Span::raw(&self.text[start..self.line_end()]);
        let mut end = start;
        for grapheme in span.styled_graphemes(Style::default()) {
            if self.cursor == end {
                break;
            }
            end += grapheme.symbol.len();
            if self.cursor <= end {
                self.cursor = end;
                break;
            }
        }
    }
    fn line_start(&self) -> usize {
        self.text[..self.cursor].rfind('\n').map_or(0, |i| i + 1)
    }
    fn line_end(&self) -> usize {
        self.text[self.cursor..]
            .find('\n')
            .map_or(self.text.len(), |i| self.cursor + i)
    }
    fn vertical(&mut self, down: bool) {
        let start = self.line_start();
        let column = self.text[start..self.cursor].width();
        let target = if down {
            let end = self.line_end();
            if end == self.text.len() {
                return;
            }
            end + 1
        } else {
            if start == 0 {
                return;
            }
            self.text[..start - 1].rfind('\n').map_or(0, |i| i + 1)
        };
        let line = self.text[target..].split('\n').next().unwrap_or("");
        let span = Span::raw(line);
        let mut cells = 0;
        self.cursor = target;
        for grapheme in span.styled_graphemes(Style::default()) {
            cells += grapheme.symbol.width();
            if cells > column {
                break;
            }
            self.cursor += grapheme.symbol.len();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }
    #[test]
    fn unicode_editing_keeps_byte_boundaries() {
        let mut e = Editor::default();
        e.insert("a界🙂e\u{301}");
        e.key(key(KeyCode::Left));
        e.key(key(KeyCode::Backspace));
        assert_eq!(e.text, "a界e\u{301}");
        e.key(key(KeyCode::Home));
        e.key(key(KeyCode::Right));
        e.key(key(KeyCode::Delete));
        assert_eq!(e.text, "ae\u{301}");
        assert!(e.text.is_char_boundary(e.cursor));
    }
    #[test]
    fn joined_emoji_and_combining_characters_are_single_edit_units() {
        let mut e = Editor::new("a👩‍💻e\u{301}".into());
        e.key(key(KeyCode::Backspace));
        assert_eq!(e.text, "a👩‍💻");
        e.key(key(KeyCode::Left));
        assert_eq!(e.cursor, 1);
        e.key(key(KeyCode::Delete));
        assert_eq!(e.text, "a");
    }
    #[test]
    fn vertical_and_paste_are_safe() {
        let mut e = Editor::new("界ab\n🙂x".into());
        e.key(key(KeyCode::Up));
        assert_eq!(&e.text[..e.cursor], "界a");
        e.insert("\x1b]52;c;abc\x07z");
        assert_eq!(e.text, "界azb\n🙂x");
    }
}
