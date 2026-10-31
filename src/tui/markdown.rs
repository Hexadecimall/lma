//! A display-only Markdown subset: no terminal protocols, HTML, or executable links.
use super::text::sanitize;
use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};
use unicode_width::UnicodeWidthStr;

#[derive(Default)]
pub struct Cache(std::collections::HashMap<usize, Entry>);
struct Entry {
    text: String,
    width: u16,
    lines: Vec<Line<'static>>,
}
impl Cache {
    pub fn clear(&mut self) {
        self.0.clear();
    }
    pub fn lines(&mut self, index: usize, text: &str, width: u16) -> Vec<Line<'static>> {
        if let Some(entry) = self.0.get(&index)
            && entry.width == width
            && entry.text == text
        {
            return entry.lines.clone();
        }
        let entry = Entry {
            text: text.into(),
            width,
            lines: render(text, width),
        };
        let lines = entry.lines.clone();
        self.0.insert(index, entry);
        lines
    }
}

fn code_style() -> Style {
    Style::default().fg(Color::Yellow)
}

pub fn render(text: &str, width: u16) -> Vec<Line<'static>> {
    let safe = sanitize(text);
    let mut lines = Vec::new();
    let mut fence: Option<(char, usize)> = None;
    for source in safe.split('\n') {
        let trimmed = source.trim_start();
        let indent = source.len() - trimmed.len();
        if let Some((marker, count)) = fence {
            let run = trimmed.chars().take_while(|c| *c == marker).count();
            if indent <= 3 && run >= count && trimmed[run..].trim().is_empty() {
                fence = None;
                continue;
            }
            let spans = vec![
                Span::styled("│ ", Style::default().fg(Color::DarkGray)),
                Span::styled(source.to_owned(), code_style()),
            ];
            lines.extend(wrap_spans(spans, width));
            continue;
        }
        let marker = trimmed.chars().next().unwrap_or(' ');
        let count = trimmed.chars().take_while(|c| *c == marker).count();
        if indent <= 3 && matches!(marker, '`' | '~') && count >= 3 {
            fence = Some((marker, count));
            let language = trimmed[count..].trim();
            lines.extend(wrap_spans(
                vec![Span::styled(
                    if language.is_empty() {
                        "┌ code".into()
                    } else {
                        format!("┌ {language}")
                    },
                    Style::default().fg(Color::DarkGray),
                )],
                width,
            ));
            continue;
        }
        let heading = trimmed.bytes().take_while(|c| *c == b'#').count();
        let (content, style) = if indent <= 3
            && (1..=6).contains(&heading)
            && trimmed.as_bytes().get(heading) == Some(&b' ')
        {
            (
                trimmed[heading + 1..].trim_end_matches(" #"),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            (source, Style::default())
        };
        let mut spans = Vec::new();
        let quote = content.trim_start();
        let (content, style) = if let Some(rest) = quote.strip_prefix('>') {
            spans.push(Span::styled("│ ", Style::default().fg(Color::Cyan)));
            (
                rest.strip_prefix(' ').unwrap_or(rest),
                style.fg(Color::Gray),
            )
        } else {
            (content, style)
        };
        let item = content.trim_start();
        let pad = content.len() - item.len();
        if let Some(rest) = ["- ", "* ", "+ "]
            .into_iter()
            .find_map(|prefix| item.strip_prefix(prefix))
        {
            spans.push(Span::raw(" ".repeat(pad)));
            spans.push(Span::styled("• ", Style::default().fg(Color::Cyan)));
            inline(rest, style, 0, &mut spans);
        } else if is_rule(item) {
            spans.push(Span::styled(
                "─".repeat(usize::from(width.max(1))),
                Style::default().fg(Color::DarkGray),
            ));
        } else {
            inline(content, style, 0, &mut spans);
        }
        lines.extend(wrap_spans(spans, width));
    }
    lines
}

fn is_rule(text: &str) -> bool {
    let mut chars = text.chars().filter(|c| *c != ' ');
    let Some(marker @ ('-' | '*' | '_')) = chars.next() else {
        return false;
    };
    let mut count = 1;
    for c in chars {
        if c != marker {
            return false;
        }
        count += 1;
    }
    count >= 3
}

fn inline(text: &str, style: Style, depth: usize, spans: &mut Vec<Span<'static>>) {
    // Bound delimiter-search work for pathological single-line model output.
    if depth >= 16 || text.len() > 16 * 1024 {
        spans.push(Span::styled(text.to_owned(), style));
        return;
    }
    let mut rest = text;
    while !rest.is_empty() {
        if let Some(escaped) = rest.strip_prefix('\\')
            && let Some(c) = escaped.chars().next().filter(|c| c.is_ascii_punctuation())
        {
            spans.push(Span::styled(c.to_string(), style));
            rest = &escaped[c.len_utf8()..];
            continue;
        }
        if rest.starts_with('`') {
            let count = rest.bytes().take_while(|c| *c == b'`').count();
            let marker = &rest[..count];
            if let Some(end) = rest[count..].find(marker) {
                spans.push(Span::styled(
                    rest[count..count + end].to_owned(),
                    style.patch(code_style()),
                ));
                rest = &rest[count + end + count..];
                continue;
            }
        }
        if rest.starts_with('[') || rest.starts_with("![") {
            let offset = usize::from(rest.starts_with('!'));
            if let Some(end) = rest[offset + 1..].find("](") {
                let label_end = offset + 1 + end;
                let target_start = label_end + 2;
                if let Some(close) = rest[target_start..].find(')') {
                    let target = &rest[target_start..target_start + close];
                    if offset == 1 {
                        spans.push(Span::styled("[image: ", style));
                    }
                    inline(
                        &rest[offset + 1..label_end],
                        style.fg(Color::Blue).add_modifier(Modifier::UNDERLINED),
                        depth + 1,
                        spans,
                    );
                    if offset == 1 {
                        spans.push(Span::styled("]", style));
                    }
                    spans.push(Span::styled(
                        format!(" ({target})"),
                        style.fg(Color::DarkGray),
                    ));
                    rest = &rest[target_start + close + 1..];
                    continue;
                }
            }
        }
        let mut consumed = false;
        for (marker, modifier) in [
            ("***", Modifier::BOLD | Modifier::ITALIC),
            ("**", Modifier::BOLD),
            ("__", Modifier::BOLD),
            ("~~", Modifier::CROSSED_OUT),
            ("*", Modifier::ITALIC),
            ("_", Modifier::ITALIC),
        ] {
            // Underscores within identifiers/paths are ordinary text, not emphasis.
            if marker.starts_with('_')
                && text[..text.len() - rest.len()]
                    .chars()
                    .next_back()
                    .is_some_and(|c| c.is_alphanumeric() || c == '_')
            {
                continue;
            }
            if let Some(body) = rest.strip_prefix(marker)
                && !body.starts_with(char::is_whitespace)
                && let Some(end) = body.find(marker).filter(|end| *end > 0)
                && !body[..end].ends_with(char::is_whitespace)
            {
                inline(&body[..end], style.add_modifier(modifier), depth + 1, spans);
                rest = &body[end + marker.len()..];
                consumed = true;
                break;
            }
        }
        if consumed {
            continue;
        }
        // Batch ordinary text into one span; do not split graphemes into cells here.
        let first = rest.chars().next().unwrap().len_utf8();
        let end = rest[first..]
            .find(['\\', '`', '[', '!', '*', '_', '~'])
            .map_or(rest.len(), |i| first + i);
        spans.push(Span::styled(rest[..end].to_owned(), style));
        rest = &rest[end..];
    }
}

/// Wrap styled graphemes together, retaining attributes and terminal cell widths.
fn wrap_spans(spans: Vec<Span<'static>>, width: u16) -> Vec<Line<'static>> {
    let width = usize::from(width.max(1));
    let line = Line::from(spans);
    let mut lines = Vec::new();
    let mut current: Vec<Span<'static>> = Vec::new();
    let mut columns = 0;
    for grapheme in line.styled_graphemes(Style::default()) {
        let cells = grapheme.symbol.width();
        if columns + cells > width && !current.is_empty() {
            lines.push(Line::from(std::mem::take(&mut current)));
            columns = 0;
        }
        let symbol = if cells > width {
            "�"
        } else {
            grapheme.symbol
        };
        if let Some(last) = current
            .last_mut()
            .filter(|span| span.style == grapheme.style)
        {
            last.content.to_mut().push_str(symbol);
        } else {
            current.push(Span::styled(symbol.to_owned(), grapheme.style));
        }
        columns += cells.min(width);
    }
    lines.push(Line::from(current));
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    fn text(lines: &[Line<'_>]) -> String {
        lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
    #[test]
    fn headings_emphasis_lists_links_and_literal_html() {
        let lines = render(
            "# Title\n**bold** and *italic* and `x_y`\n- item\n1. numbered\n> quote\n[label](https://example.org)\n<script>literal</script>",
            100,
        );
        let output = text(&lines);
        assert!(output.contains("Title\nbold and italic and x_y\n• item\n1. numbered\n│ quote"));
        assert!(output.contains("label (https://example.org)"));
        assert!(output.contains("<script>literal</script>"));
        assert!(
            lines[0].spans[0]
                .style
                .add_modifier
                .contains(Modifier::BOLD)
        );
        assert!(
            lines[1]
                .spans
                .iter()
                .any(|span| span.content == "bold"
                    && span.style.add_modifier.contains(Modifier::BOLD))
        );
        assert!(
            lines[1].spans.iter().any(|span| span.content == "italic"
                && span.style.add_modifier.contains(Modifier::ITALIC))
        );
    }
    #[test]
    fn identifiers_unmatched_syntax_and_pathological_lines_stay_readable() {
        assert_eq!(
            text(&render("snake_case_name /path/to_some_file.rs", 100)),
            "snake_case_name /path/to_some_file.rs"
        );
        let long = "[".repeat(32 * 1024);
        assert_eq!(text(&render(&long, 80)).replace('\n', ""), long);
        assert_eq!(text(&render("**bold** _italic_", 100)), "bold italic");
    }
    #[test]
    fn code_is_literal_and_open_streaming_fences_are_supported() {
        let output = text(&render(
            "```rust\nlet x = **not bold**;\n```\n~~~\n界🙂\n",
            80,
        ));
        assert!(output.contains("┌ rust\n│ let x = **not bold**;"));
        assert!(output.contains("┌ code\n│ 界🙂"));
        assert!(!output.contains("```"));
        assert_eq!(text(&render("a **unfinished", 80)), "a **unfinished");
        assert_eq!(text(&render("\\*literal\\*", 80)), "*literal*");
    }
    #[test]
    fn sanitizes_model_protocols_and_wraps_unicode_without_losing_styles() {
        let source = "**界👩‍💻e\u{301}abcdef**\n\u{1b}]52;c;payload\u{7}safe\u{202e}";
        for width in [1, 2, 3, 8, 80] {
            let lines = render(source, width);
            assert!(lines.iter().all(|line| line.width() <= usize::from(width)));
            let output = text(&lines);
            assert!(!output.contains("payload"));
            assert!(!output.contains('\u{1b}'));
            assert!(!output.contains('\u{202e}'));
        }
        let lines = render("**abcdefgh**", 4);
        assert_eq!(text(&lines), "abcd\nefgh");
        assert!(lines.iter().all(|line| {
            line.spans
                .iter()
                .all(|span| span.style.add_modifier.contains(Modifier::BOLD))
        }));
    }
    #[test]
    fn cache_invalidates_on_content_width_and_reset() {
        let mut cache = Cache::default();
        assert_eq!(text(&cache.lines(0, "**one**", 80)), "one");
        assert_eq!(text(&cache.lines(0, "**two**", 80)), "two");
        assert_eq!(text(&cache.lines(0, "**two**", 2)), "tw\no");
        cache.clear();
        assert!(cache.0.is_empty());
    }
}
