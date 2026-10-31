use super::{
    markdown,
    state::{App, Focus, Panel},
    text::{one_line, sanitize, wrap},
};
use ratatui::{
    Frame,
    layout::{Constraint, Layout},
    style::{Color, Modifier, Style},
    text::Line,
    widgets::{Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState},
};

fn add(lines: &mut Vec<Line<'static>>, text: &str, width: u16, style: Style) {
    let safe = sanitize(text);
    let (wrapped, _) = wrap(&safe, width, usize::MAX);
    lines.extend(wrapped.into_iter().map(|line| Line::styled(line, style)));
}

fn add_reply(lines: &mut Vec<Line<'static>>, text: &str, width: u16, markdown_enabled: bool) {
    if markdown_enabled {
        lines.extend(markdown::render(text, width));
    } else {
        add(lines, text, width, Style::default());
    }
}

pub fn draw(frame: &mut Frame, app: &mut App) {
    let areas = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(1),
        Constraint::Length(6),
        Constraint::Length(3),
    ])
    .split(frame.area());
    let header = vec![
        Line::styled(
            "lma / local workspace",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Line::from(format!(
            "{} · {} · context {} · GPU layers {} requested · /logs for native placement",
            app.model,
            if app.raw { "raw" } else { "chat" },
            app.config.context,
            app.config.gpu_layers
        )),
    ];
    frame.render_widget(Paragraph::new(header), areas[0]);
    let history_title = match app.panel {
        Some(Panel::Help) => " Help ",
        Some(Panel::Logs) => " Native logs (bounded tail) ",
        None => {
            if app.scroll == 0 {
                " Conversation · live "
            } else {
                " Conversation · scrolled · Ctrl+End for live "
            }
        }
    };
    let history_block = Block::default()
        .borders(Borders::TOP | Borders::RIGHT)
        .title(history_title)
        .border_style(
            Style::default().fg(if app.focus == Focus::History || app.panel.is_some() {
                Color::Cyan
            } else {
                Color::DarkGray
            }),
        );
    let inner = history_block.inner(areas[1]);
    let mut lines = Vec::new();
    if app.panel.is_some() {
        add(&mut lines, &app.panel_text(), inner.width, Style::default());
    } else {
        if app.history.is_empty() && app.busy.is_none() {
            add(
                &mut lines,
                "What would you like to work on?",
                inner.width,
                Style::default().fg(Color::DarkGray),
            );
            add(
                &mut lines,
                "Text chat, not a tool-running agent. /help for commands.",
                inner.width,
                Style::default().fg(Color::DarkGray),
            );
        }
        for (index, message) in app.history.iter().enumerate() {
            lines.push(Line::styled(
                if message.role == "user" {
                    "You"
                } else {
                    "Assistant"
                },
                Style::default()
                    .fg(if message.role == "user" {
                        Color::Cyan
                    } else {
                        Color::Green
                    })
                    .add_modifier(Modifier::BOLD),
            ));
            if message.role == "assistant" && app.markdown_enabled {
                lines.extend(app.markdown_cache.lines(index, &message.text, inner.width));
            } else {
                add(&mut lines, &message.text, inner.width, Style::default());
            }
            for image in &message.images {
                add(
                    &mut lines,
                    &format!("[image: {}]", one_line(&image.name)),
                    inner.width,
                    Style::default().fg(Color::DarkGray),
                );
            }
            lines.push(Line::from(""));
        }
        if app.busy.is_some() {
            lines.push(Line::styled("You", Style::default().fg(Color::Cyan)));
            add(&mut lines, &app.editor.text, inner.width, Style::default());
            lines.push(Line::styled(
                "Assistant · generating…",
                Style::default().fg(Color::Yellow),
            ));
            add_reply(&mut lines, &app.partial, inner.width, app.markdown_enabled);
        } else if !app.partial.is_empty() {
            lines.push(Line::styled(
                format!(
                    "Assistant · partial · {}",
                    app.partial_status.unwrap_or("interrupted · not in history")
                ),
                Style::default().fg(Color::Yellow),
            ));
            add_reply(&mut lines, &app.partial, inner.width, app.markdown_enabled);
        }
    }
    let total_lines = lines.len();
    let start = app.viewport(total_lines, usize::from(inner.height));
    let visible = lines
        .into_iter()
        .skip(start)
        .take(inner.height as usize)
        .collect::<Vec<_>>();
    frame.render_widget(history_block, areas[1]);
    frame.render_widget(Paragraph::new(visible), inner);
    if total_lines > usize::from(inner.height) {
        let mut scrollbar = ScrollbarState::new(total_lines)
            .position(start)
            .viewport_content_length(usize::from(inner.height));
        frame.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None),
            areas[1],
            &mut scrollbar,
        );
    }

    let composer = Block::default()
        .borders(Borders::ALL)
        .title(format!(
            " {}{} ",
            if app.raw { "Raw prompt" } else { "Message" },
            if app.images.is_empty() {
                String::new()
            } else {
                format!(" · {} images", app.images.len())
            }
        ))
        .border_style(Style::default().fg(
            if app.focus == Focus::Composer && app.panel.is_none() {
                Color::Cyan
            } else {
                Color::DarkGray
            },
        ));
    let inner = composer.inner(areas[2]);
    let (input, (row, col)) = wrap(&app.editor.text, inner.width, app.editor.cursor);
    let top = row.saturating_sub(inner.height.saturating_sub(1) as usize);
    let visible = input
        .into_iter()
        .skip(top)
        .take(inner.height as usize)
        .map(Line::from)
        .collect::<Vec<_>>();
    frame.render_widget(composer, areas[2]);
    frame.render_widget(Paragraph::new(visible), inner);
    if app.focus == Focus::Composer
        && app.panel.is_none()
        && app.busy.is_none()
        && inner.width > 0
        && inner.height > 0
    {
        frame.set_cursor_position((
            inner.x + col.min(inner.width - 1),
            inner.y + (row - top) as u16,
        ));
    }
    let status = match app.busy {
        Some(started) => format!(
            "Generating · {:.1}s · reply limit {} · Esc / Ctrl+C cancel",
            started.elapsed().as_secs_f32(),
            app.token_limit_label()
        ),
        None => format!(
            "{} · reply limit {} · temp {} · {}",
            if app.raw { "Raw" } else { "Chat" },
            app.token_limit_label(),
            app.temperature,
            if app.usage.is_empty() {
                app.partial_status.unwrap_or("Ready")
            } else {
                &app.usage
            }
        ),
    };
    let footer = vec![
        Line::styled(
            status,
            Style::default().fg(if app.busy.is_some() {
                Color::Yellow
            } else {
                Color::DarkGray
            }),
        ),
        Line::from(one_line(&app.notice)),
        Line::styled(
            "Enter send · Shift+Enter newline (Alt+Enter fallback) · wheel/PgUp/PgDn scroll · /help",
            Style::default().fg(Color::DarkGray),
        ),
    ];
    frame.render_widget(Paragraph::new(footer), areas[3]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ServeConfig;
    use crate::tui::state::Message;
    use ratatui::{Terminal, backend::TestBackend};
    #[test]
    fn renders_unicode_and_markdown_as_safe_display_text() {
        let mut app = App::new(
            ServeConfig {
                model: "model\x1b]0;owned\x07.gguf".into(),
                ..Default::default()
            },
            false,
        );
        app.history.push(Message {
            role: "assistant",
            text: "Hello 界🙂\n```rust\nfn main() {}\n```\x1b]52;c;payload\x07".into(),
            images: vec![],
        });
        app.editor.insert("你好\n🙂");
        let mut terminal = Terminal::new(TestBackend::new(90, 26)).unwrap();
        terminal.draw(|f| draw(f, &mut app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains("fn main() {}"));
        assert!(text.contains("model.gguf"));
        assert!(!text.contains("payload"));
        assert!(!text.contains("owned"));
    }
    #[test]
    fn streamed_text_is_visible_before_completion_and_abort_is_not_a_final_reply() {
        let mut app = App::new(ServeConfig::default(), false);
        app.editor.insert("question");
        app.begin();
        app.partial.push_str("first token 界");
        let mut terminal = Terminal::new(TestBackend::new(100, 26)).unwrap();
        terminal.draw(|f| draw(f, &mut app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains("Assistant · generating"));
        assert!(text.contains("first token 界"));
        assert!(app.history.is_empty());
        app.cancel();
        terminal.draw(|f| draw(f, &mut app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains("partial · cancelled · not in history"));
        assert!(text.contains("first token 界"));
        assert!(!text.contains("Completed"));
    }
    #[test]
    fn small_terminal_and_scrolling_never_panic() {
        let mut app = App::new(ServeConfig::default(), false);
        app.scroll = usize::MAX;
        for (w, h) in [(1, 1), (10, 5), (40, 12)] {
            let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
            terminal.draw(|f| draw(f, &mut app)).unwrap();
        }
    }
}
