//! Read-only listing uses the same bounded discovery as the GUI and launcher.
use std::io::{self, Write};

use crate::{
    config::ListConfig,
    models::{self, DiscoveryOptions, ModelCatalog},
    terminal_style::{TerminalStyle, safe},
};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

pub fn run(config: ListConfig) -> Result<(), String> {
    let mut roots = models::discovery::default_local_dirs();
    roots.extend(config.local_dirs);
    let mut catalog = models::discover(DiscoveryOptions {
        local_dirs: roots,
        ollama_dir: config.ollama_dir,
    });
    let filter = config.filter.to_lowercase();
    catalog.models.retain(|model| {
        model.name.to_lowercase().contains(&filter) || model.model.to_lowercase().contains(&filter)
    });
    catalog.models.sort_by(|a, b| {
        a.name
            .to_lowercase()
            .cmp(&b.name.to_lowercase())
            .then(a.id.cmp(&b.id))
    });
    let output = if config.json {
        serde_json::to_string_pretty(&catalog).map_err(|error| error.to_string())? + "\n"
    } else {
        let warnings = TerminalStyle::stderr();
        for warning in &catalog.warnings {
            eprintln!("{}", warnings.warning(warning));
        }
        let style = TerminalStyle::stdout();
        let width = style.is_terminal().then(|| {
            crossterm::terminal::size()
                .ok()
                .map(|(columns, _)| usize::from(columns).max(1))
                .unwrap_or(80)
        });
        table(&catalog, style, width)
    };
    match io::stdout().lock().write_all(output.as_bytes()) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        Err(error) => Err(format!("cannot print models: {error}")),
    }
}

fn size(bytes: u64) -> String {
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let (factor, unit) = if bytes < 1024 * 1024 {
        (1024_f64, "KiB")
    } else if bytes < 1024 * 1024 * 1024 {
        (1024_f64.powi(2), "MiB")
    } else {
        (1024_f64.powi(3), "GiB")
    };
    format!("{:.1} {unit}", bytes as f64 / factor)
}

type Row = (String, String, String, String);

fn table(catalog: &ModelCatalog, style: TerminalStyle, width: Option<usize>) -> String {
    if catalog.models.is_empty() {
        let message = "No installed models found. Try `lma f MODEL` or `lma l -dr DIR`.";
        return match width {
            Some(width) => wrapped(message, width).join("\n") + "\n",
            None => format!("{message}\n"),
        };
    }
    let rows: Vec<_> = catalog
        .models
        .iter()
        .map(|m| {
            (
                safe(&m.name),
                safe(&m.source),
                size(m.size_bytes),
                safe(&m.model),
            )
        })
        .collect();
    if let Some(width) = width {
        return terminal_table(&rows, style, width);
    }
    // Keep the original, untruncated layout for scripts and redirected output.
    let name_width = rows.iter().map(|r| r.0.width()).max().unwrap_or(0).max(4);
    let source_width = rows.iter().map(|r| r.1.width()).max().unwrap_or(0).max(6);
    let size_width = rows.iter().map(|r| r.2.width()).max().unwrap_or(0).max(4);
    let mut text = String::new();
    for (name, source, size, model) in std::iter::once((
        "NAME".into(),
        "SOURCE".into(),
        "SIZE".into(),
        "MODEL".into(),
    ))
    .chain(rows)
    {
        text.push_str(&name);
        text.push_str(&" ".repeat(name_width - name.width() + 2));
        text.push_str(&source);
        text.push_str(&" ".repeat(source_width - source.width() + 2));
        text.push_str(&" ".repeat(size_width - size.width()));
        text.push_str(&size);
        text.push_str("  ");
        text.push_str(&model);
        text.push('\n');
    }
    text
}

fn terminal_table(rows: &[Row], style: TerminalStyle, width: usize) -> String {
    let width = width.max(1);
    let mut text = String::new();
    if width < 76 {
        for line in wrapped("INSTALLED MODELS", width) {
            text.push_str(&style.heading(&line));
            text.push('\n');
        }
        for (name, source, size, model) in rows {
            text.push('\n');
            for line in wrapped(name, width) {
                text.push_str(&style.accent(&line));
                text.push('\n');
            }
            for line in wrapped(&format!("{source} · {size}"), width) {
                text.push_str(&style.muted(&line));
                text.push('\n');
            }
            for line in wrapped(&format!("MODEL  {model}"), width) {
                text.push_str(&line);
                text.push('\n');
            }
        }
        return text;
    }

    let source_width = rows
        .iter()
        .map(|r| r.1.width())
        .max()
        .unwrap_or(6)
        .clamp(6, 16);
    let size_width = rows.iter().map(|r| r.2.width()).max().unwrap_or(4).max(4);
    let available = width - source_width - size_width - 6;
    let name_width = rows
        .iter()
        .map(|r| r.0.width())
        .max()
        .unwrap_or(4)
        .clamp(4, (available / 3).max(4));
    let model_width = available - name_width;
    let header = format!(
        "{}  {}  {}  {}",
        padded("NAME", name_width, false),
        padded("SOURCE", source_width, false),
        padded("SIZE", size_width, true),
        "MODEL",
    );
    text.push_str(&style.heading(&header));
    text.push('\n');
    for (name, source, size, model) in rows {
        text.push_str(&style.accent(&padded(name, name_width, false)));
        text.push_str("  ");
        text.push_str(&padded(source, source_width, false));
        text.push_str("  ");
        text.push_str(&padded(size, size_width, true));
        text.push_str("  ");
        text.push_str(&clipped(model, model_width));
        text.push('\n');
    }
    text
}

fn padded(text: &str, width: usize, right: bool) -> String {
    let text = clipped(text, width);
    let padding = " ".repeat(width.saturating_sub(text.width()));
    if right {
        padding + &text
    } else {
        text + &padding
    }
}

fn clipped(text: &str, width: usize) -> String {
    if text.width() <= width {
        return text.into();
    }
    if width == 0 {
        return String::new();
    }
    let mut result = String::new();
    let mut used = 0;
    for c in text.chars() {
        let columns = c.width().unwrap_or(0);
        if used + columns > width - 1 {
            break;
        }
        result.push(c);
        used += columns;
    }
    result.push('…');
    result
}

fn wrapped(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut lines = Vec::new();
    let mut line = String::new();
    let mut used = 0;
    for c in text.chars() {
        let columns = c.width().unwrap_or(0);
        if used + columns > width && !line.is_empty() {
            lines.push(std::mem::take(&mut line));
            used = 0;
        }
        // A double-width glyph cannot fit in a one-column terminal.
        if columns > width {
            line.push('?');
            used += 1;
        } else {
            line.push(c);
            used += columns;
        }
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_layouts_fit_narrow_and_wide_unicode_columns() {
        let rows = vec![(
            "模型 with a very long name".into(),
            "lm-studio".into(),
            "12.3 GiB".into(),
            "/a/very/long/模型/path/with/more/components/model.gguf".into(),
        )];
        // Unit-test the layout independently of the test runner's terminal.
        for width in [1, 2, 20, 40, 75, 76, 80, 120] {
            let plain = terminal_table(
                &rows,
                TerminalStyle::for_environment(false, None, None),
                width,
            );
            assert!(
                plain.lines().all(|line| line.width() <= width),
                "width {width}: {plain}"
            );
            if width < 76 && width >= 20 {
                assert!(plain.contains("MODEL"));
                assert!(!plain.contains('…'));
            }
        }
        assert_eq!(clipped("模型abc", 4), "模…");
        assert_eq!(clipped("a\u{301}bc", 2), "a\u{301}…");
    }

    #[test]
    fn file_sizes_are_binary_and_names_cannot_escape_the_terminal() {
        assert_eq!(size(4), "4 B");
        assert_eq!(size(1024), "1.0 KiB");
        assert_eq!(size(1024 * 1024), "1.0 MiB");
        assert_eq!(size(1024 * 1024 * 1024), "1.0 GiB");
        let escaped = safe("model\x1b[2J\n\u{202e}.gguf");
        assert!(!escaped.chars().any(char::is_control));
        assert!(!escaped.contains('\u{202e}'));
        assert!(escaped.contains("\\u{1b}"));
        assert!(
            table(&ModelCatalog::default(), TerminalStyle::stdout(), None)
                .starts_with("No installed models")
        );
    }
}
