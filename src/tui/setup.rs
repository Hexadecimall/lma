use super::{
    editor::Editor,
    terminal::Screen,
    text::{one_line, sanitize, wrap},
};
use crate::{
    config::{LaunchConfig, LaunchTarget},
    models::discovery::{self, DiscoveredModel, DiscoveryOptions, ModelCatalog},
};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::Line,
    widgets::{Block, Borders, Paragraph},
};
use std::{path::PathBuf, time::Duration};

const APIS: &[&str] = &["all", "openai", "ollama", "anthropic"];
const CLIENTS: &[&str] = crate::config::CLIENT_PRESETS;

#[derive(Clone, Copy)]
enum Row {
    Model,
    Target,
    Gpu,
    Context,
    Api,
    Client,
    Start,
}
const CHAT_ROWS: &[Row] = &[Row::Model, Row::Gpu, Row::Context, Row::Start];
const LAUNCH_ROWS: &[Row] = &[Row::Model, Row::Target, Row::Gpu, Row::Context, Row::Start];
const HOST_ROWS: &[Row] = &[
    Row::Model,
    Row::Gpu,
    Row::Context,
    Row::Api,
    Row::Client,
    Row::Start,
];

pub struct Setup {
    pub config: LaunchConfig,
    pub catalog: ModelCatalog,
    pub row: usize,
    pub edit: Option<Editor>,
    pub picker: Option<usize>,
    pub notice: String,
}

fn cycle(value: &mut String, choices: &[&str], delta: isize) {
    let index = choices.iter().position(|c| *c == value).unwrap_or(0);
    *value = choices[(index as isize + delta).rem_euclid(choices.len() as isize) as usize].into();
}

fn selection_reference(model: &DiscoveredModel) -> PathBuf {
    if model.ollama {
        PathBuf::from(format!("ollama:{}", model.model))
    } else if model.id == format!("lma:{}", model.name) {
        PathBuf::from(&model.id)
    } else {
        PathBuf::from(&model.model)
    }
}

fn model_label(model: &DiscoveredModel) -> String {
    let source = match model.source.to_ascii_lowercase().as_str() {
        "local" | "lma" => "lma",
        "ollama" => "Ollama",
        "lmstudio" | "lm-studio" | "lm_studio" | "lm studio" => "LM Studio",
        _ => &model.source,
    };
    one_line(&format!("[{source}] {}", model.name))
}

impl Setup {
    pub fn new(config: LaunchConfig, catalog: ModelCatalog) -> Self {
        let mut setup = Self {
            config,
            catalog,
            row: 0,
            edit: None,
            picker: None,
            notice: String::new(),
        };
        if setup.config.serve.model.as_os_str().is_empty() && !setup.catalog.models.is_empty() {
            setup.select(0);
        }
        setup
    }
    fn rows(&self) -> &'static [Row] {
        match self.config.mode.as_str() {
            "host" => HOST_ROWS,
            "launch" => LAUNCH_ROWS,
            _ => CHAT_ROWS,
        }
    }
    fn current(&self) -> Option<usize> {
        self.catalog.models.iter().position(|m| {
            selection_reference(m) == self.config.serve.model
                && (!m.ollama
                    || self.config.serve.ollama_dir.is_none()
                    || m.ollama_dir == self.config.serve.ollama_dir)
        })
    }
    fn select(&mut self, index: usize) {
        let model = &self.catalog.models[index];
        self.config.serve.model = selection_reference(model);
        self.config.serve.model_id = None;
        self.config.serve.ollama_dir = model.ollama_dir.clone().or(self
            .config
            .serve
            .ollama_dir
            .clone());
        self.config.serve.projector = model.projector.clone();
    }
    fn mark_explicit_load(&mut self, name: &str) {
        if !self
            .config
            .serve
            .explicit_load
            .iter()
            .any(|key| key == name)
        {
            self.config.serve.explicit_load.push(name.into());
        }
    }
    fn open_picker(&mut self) {
        self.picker = Some(self.current().unwrap_or(0));
        self.notice.clear();
    }
    pub fn key(&mut self, key: KeyEvent) -> Option<bool> {
        if key.modifiers.contains(KeyModifiers::CONTROL)
            && matches!(key.code, KeyCode::Char('c' | 'q'))
        {
            return Some(false);
        }
        if let Some(index) = self.picker {
            let count = self.catalog.models.len();
            match key.code {
                KeyCode::Esc => self.picker = None,
                KeyCode::Enter if count > 0 => {
                    self.select(index);
                    self.picker = None;
                }
                KeyCode::Up | KeyCode::BackTab if count > 0 => {
                    self.picker = Some((index + count - 1) % count)
                }
                KeyCode::Down | KeyCode::Tab if count > 0 => {
                    self.picker = Some((index + 1) % count)
                }
                KeyCode::Home if count > 0 => self.picker = Some(0),
                KeyCode::End if count > 0 => self.picker = Some(count - 1),
                KeyCode::PageUp if count > 0 => self.picker = Some(index.saturating_sub(10)),
                KeyCode::PageDown if count > 0 => self.picker = Some((index + 10).min(count - 1)),
                _ => {}
            }
            return None;
        }
        let row = self.rows()[self.row];
        if let Some(editor) = &mut self.edit {
            match key.code {
                KeyCode::Esc => self.edit = None,
                KeyCode::Enter => {
                    let value = editor.text.trim().to_owned();
                    let result = match row {
                        Row::Gpu => value.parse::<u32>().map(|n| self.config.serve.gpu_layers = n).map_err(|_| "GPU layers must be 0 (CPU) or a positive integer; 999 requests all layers"),
                        Row::Context => value.parse::<u32>().ok().filter(|n| *n >= 64).map(|n| self.config.serve.context = n).ok_or("Context must be an integer of at least 64 tokens"),
                        _ => Err("Only GPU layers and context accept numeric input"),
                    };
                    match result {
                        Ok(()) => {
                            match row {
                                Row::Gpu => self.mark_explicit_load("num_gpu"),
                                Row::Context => self.mark_explicit_load("num_ctx"),
                                _ => {}
                            }
                            self.edit = None;
                            self.notice.clear();
                        }
                        Err(e) => self.notice = e.into(),
                    }
                }
                _ => editor.key(key),
            }
            return None;
        }
        let count = self.rows().len();
        match key.code {
            KeyCode::Esc => return Some(false),
            KeyCode::Up | KeyCode::BackTab => self.row = (self.row + count - 1) % count,
            KeyCode::Down | KeyCode::Tab => self.row = (self.row + 1) % count,
            KeyCode::Left => self.change(-1),
            KeyCode::Right | KeyCode::Char(' ') => self.change(1),
            KeyCode::Enter => match row {
                Row::Model => self.open_picker(),
                Row::Gpu => self.edit = Some(Editor::new(self.config.serve.gpu_layers.to_string())),
                Row::Context => {
                    self.edit = Some(Editor::new(self.config.serve.context.to_string()))
                }
                Row::Start if self.config.serve.model.as_os_str().is_empty() => {
                    self.notice =
                        "Choose an installed model first, or relaunch with -p FILE.".into();
                }
                Row::Start => return Some(true),
                _ => self.change(1),
            },
            _ => {}
        }
        None
    }
    fn change(&mut self, delta: isize) {
        match self.rows()[self.row] {
            Row::Model => self.open_picker(),
            Row::Target => {
                let choices = LaunchTarget::ALL;
                let index = choices
                    .iter()
                    .position(|target| *target == self.config.target)
                    .unwrap_or(0);
                self.config.target =
                    choices[(index as isize + delta).rem_euclid(choices.len() as isize) as usize];
            }
            Row::Api => cycle(&mut self.config.serve.api, APIS, delta),
            Row::Client => cycle(&mut self.config.serve.client, CLIENTS, delta),
            Row::Gpu => {
                self.config.serve.gpu_layers = if self.config.serve.gpu_layers == 0 {
                    999
                } else {
                    0
                };
                self.mark_explicit_load("num_gpu");
            }
            Row::Context => {
                let values = [512, 1024, 2048, 4096, 8192, 16384, 32768];
                let index = values
                    .iter()
                    .position(|n| *n == self.config.serve.context)
                    .unwrap_or(3);
                self.config.serve.context =
                    values[(index as isize + delta).rem_euclid(values.len() as isize) as usize];
                self.mark_explicit_load("num_ctx");
            }
            _ => {}
        }
    }
}

pub fn run(config: LaunchConfig) -> Result<Option<LaunchConfig>, String> {
    let catalog = discovery::discover(DiscoveryOptions {
        local_dirs: discovery::default_local_dirs(),
        ollama_dir: config.serve.ollama_dir.clone(),
    });
    let mut setup = Setup::new(config, catalog);
    let mut screen = Screen::enter()?;
    loop {
        screen
            .draw(|f| draw(f, &setup))
            .map_err(|e| format!("cannot draw setup: {e}"))?;
        if event::poll(Duration::from_millis(100)).map_err(|e| e.to_string())? {
            match event::read().map_err(|e| e.to_string())? {
                Event::Key(key) if key.kind != KeyEventKind::Release => {
                    if let Some(start) = setup.key(key) {
                        return Ok(start.then_some(setup.config));
                    }
                }
                Event::Paste(text) => {
                    if let Some(editor) = &mut setup.edit {
                        editor.insert(&one_line(&text));
                    }
                }
                _ => {}
            }
        }
    }
}

fn draw_picker(frame: &mut Frame, area: Rect, setup: &Setup, selected: usize) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Installed models ");
    let inner = block.inner(area);
    let height = inner.height as usize;
    let start = selected.saturating_sub(height.saturating_sub(1));
    let lines: Vec<Line> = if setup.catalog.models.is_empty() {
        vec![Line::from(
            "No installed models found. Use -p FILE for an explicit local model.",
        )]
    } else {
        setup
            .catalog
            .models
            .iter()
            .enumerate()
            .skip(start)
            .take(height)
            .map(|(i, model)| {
                Line::styled(
                    format!(
                        "{} {}",
                        if i == selected { "›" } else { " " },
                        model_label(model)
                    ),
                    if i == selected {
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default()
                    },
                )
            })
            .collect()
    };
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

pub fn draw(frame: &mut Frame, setup: &Setup) {
    let areas = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(5),
        Constraint::Length(5),
    ])
    .split(frame.area());
    frame.render_widget(
        Paragraph::new(
            "lma / local workspace\nChoose settings before loading weights. No downloads.",
        )
        .style(Style::default().fg(Color::Cyan)),
        areas[0],
    );
    let config = &setup.config;
    if let Some(selected) = setup.picker {
        draw_picker(frame, areas[1], setup, selected);
    } else {
        let lines: Vec<Line> = setup
            .rows()
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let value = match row {
                    Row::Model => format!(
                        "Model     {}",
                        setup
                            .current()
                            .map(|n| model_label(&setup.catalog.models[n]))
                            .unwrap_or_else(|| {
                                if config.serve.model.as_os_str().is_empty() {
                                    "No installed models — Enter to browse".into()
                                } else {
                                    format!(
                                        "Path override: {}",
                                        one_line(&config.serve.model.to_string_lossy())
                                    )
                                }
                            })
                    ),
                    Row::Target => format!("Target    {}", config.target.label()),
                    Row::Gpu => format!(
                        "GPU       {} layers{}",
                        config.serve.gpu_layers,
                        if config.serve.gpu_layers == 0 {
                            " (CPU)"
                        } else {
                            " (requested, not guaranteed placement)"
                        }
                    ),
                    Row::Context => format!("Context   {} tokens", config.serve.context),
                    Row::Api => format!("API       {}", config.serve.api),
                    Row::Client => format!("Client    {}", config.serve.client),
                    Row::Start if config.mode == "launch" => {
                        format!("Start     Launch {}", config.target.label())
                    }
                    Row::Start => "Start     Load selected model".into(),
                };
                Line::styled(
                    one_line(&format!(
                        "{} {value}",
                        if i == setup.row { "›" } else { " " }
                    )),
                    if i == setup.row {
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default()
                    },
                )
            })
            .collect();
        let block = Block::default().borders(Borders::TOP);
        let inner = block.inner(areas[1]);
        frame.render_widget(Paragraph::new(lines).block(block), areas[1]);
        if let Some(editor) = &setup.edit
            && inner.height > 0
            && inner.width > 6
        {
            let width = inner.width - 6;
            let (wrapped, (row, col)) = wrap(&editor.text, width, editor.cursor);
            let area = Rect::new(inner.x, inner.y + inner.height - 1, inner.width, 1);
            frame.render_widget(
                Paragraph::new(format!("Edit: {}", wrapped[row]))
                    .style(Style::default().fg(Color::Yellow)),
                area,
            );
            frame.set_cursor_position((area.x + 6 + col.min(width - 1), area.y));
        }
    }
    let mut footer = vec![
        Line::from(if setup.picker.is_some() {
            "↑/↓ or Tab browse · Home/End · PgUp/PgDn · Enter choose · Esc back · Ctrl+Q quit"
        } else if setup.edit.is_some() {
            "←/→ Home/End edit · Enter apply · Esc cancel edit · Ctrl+C quit"
        } else {
            "↑/↓ or Tab select · Enter browse/edit/start · ←/→ change · Esc quit"
        }),
        Line::styled(sanitize(&setup.notice), Style::default().fg(Color::Yellow)),
    ];
    if let Some(warning) = setup.catalog.warnings.first() {
        footer.push(Line::from(format!("Discovery: {}", one_line(warning))));
    }
    footer.push(Line::from(format!(
        "{} installed models · projector: {}",
        setup.catalog.models.len(),
        if config.serve.projector.is_some() {
            "yes"
        } else {
            "none (Ollama may resolve one)"
        }
    )));
    if config.mode == "host" {
        footer.push(Line::from(
            "Clients execute tools. Presets print connection settings; they do not launch agents.",
        ));
    }
    frame.render_widget(Paragraph::new(footer), areas[2]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};
    fn setup() -> Setup {
        Setup::new(
            LaunchConfig {
                serve: Default::default(),
                mode: "chat".into(),
                target: LaunchTarget::default(),
                interactive: true,
            },
            ModelCatalog::default(),
        )
    }
    fn key(s: &mut Setup, code: KeyCode) -> Option<bool> {
        s.key(KeyEvent::new(code, KeyModifiers::NONE))
    }
    fn model(name: &str, source: &str) -> DiscoveredModel {
        DiscoveredModel {
            id: format!("{source}-file:{name}"),
            name: name.into(),
            source: source.into(),
            model: format!("/models/{name}.gguf"),
            ollama: false,
            ollama_dir: None,
            projector: None,
            size_bytes: 4,
        }
    }
    fn rendered(s: &Setup) -> String {
        let mut terminal = Terminal::new(TestBackend::new(120, 24)).unwrap();
        terminal.draw(|f| draw(f, s)).unwrap();
        terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect()
    }
    #[test]
    fn model_enter_opens_a_source_labelled_list_not_an_editor() {
        let mut s = Setup::new(
            setup().config,
            ModelCatalog {
                models: vec![
                    model("one", "lma"),
                    model("two", "ollama"),
                    model("three", "lm-studio"),
                ],
                warnings: vec![],
            },
        );
        key(&mut s, KeyCode::Enter);
        assert!(s.edit.is_none());
        assert_eq!(s.picker, Some(0));
        let text = rendered(&s);
        assert!(text.contains("Installed models"));
        assert!(text.contains("[lma] one"));
        assert!(text.contains("[Ollama] two"));
        assert!(text.contains("[LM Studio] three"));
        let original = s.config.serve.model.clone();
        key(&mut s, KeyCode::Down);
        assert_eq!(s.config.serve.model, original);
        key(&mut s, KeyCode::Esc);
        assert!(s.picker.is_none());
        assert_eq!(s.config.serve.model, original);
        key(&mut s, KeyCode::Enter);
        key(&mut s, KeyCode::End);
        key(&mut s, KeyCode::Enter);
        assert_eq!(s.config.serve.model, PathBuf::from("/models/three.gguf"));
        assert!(s.picker.is_none());
    }
    #[test]
    fn ollama_selection_stays_explicit_when_a_local_filename_matches_its_tag() {
        let directory = tempfile::tempdir().unwrap();
        let mut local = model("ambiguous:latest", "lma");
        local.model = directory
            .path()
            .join("ambiguous:latest")
            .to_string_lossy()
            .into_owned();
        let mut ollama = model("ambiguous:latest", "ollama");
        ollama.model = "ambiguous:latest".into();
        ollama.ollama = true;
        ollama.ollama_dir = Some(directory.path().join("store"));
        ollama.projector = Some(directory.path().join("projector.gguf"));
        let mut s = Setup::new(
            setup().config,
            ModelCatalog {
                models: vec![local.clone(), ollama.clone()],
                warnings: vec![],
            },
        );
        key(&mut s, KeyCode::Enter);
        key(&mut s, KeyCode::Down);
        key(&mut s, KeyCode::Enter);
        assert_eq!(
            s.config.serve.model,
            PathBuf::from("ollama:ambiguous:latest")
        );
        assert_eq!(s.config.serve.ollama_dir, ollama.ollama_dir);
        assert_eq!(s.config.serve.projector, ollama.projector);
        key(&mut s, KeyCode::Enter);
        key(&mut s, KeyCode::Home);
        key(&mut s, KeyCode::Enter);
        assert_eq!(s.config.serve.model, PathBuf::from(local.model));
        assert!(s.config.serve.projector.is_none());
    }
    #[test]
    fn registry_profiles_sharing_weights_keep_distinct_selection_references() {
        let mut first = model("team/first:latest", "lma");
        first.id = format!("lma:{}", first.name);
        first.model = "/models/shared.gguf".into();
        let mut second = model("team/second:latest", "lma");
        second.id = format!("lma:{}", second.name);
        second.model = first.model.clone();
        let mut s = Setup::new(
            setup().config,
            ModelCatalog {
                models: vec![first.clone(), second.clone()],
                warnings: vec![],
            },
        );
        assert_eq!(s.config.serve.model, PathBuf::from(&first.id));
        assert_eq!(s.current(), Some(0));
        assert!(s.config.serve.explicit_load.is_empty());
        key(&mut s, KeyCode::Enter);
        key(&mut s, KeyCode::Down);
        key(&mut s, KeyCode::Enter);
        assert_eq!(s.config.serve.model, PathBuf::from(&second.id));
        assert_eq!(s.current(), Some(1));
        assert!(s.config.serve.ollama_dir.is_none());
        assert_eq!(s.catalog.models[0].model, "/models/shared.gguf");
        assert_eq!(s.catalog.models[1].model, "/models/shared.gguf");
        key(&mut s, KeyCode::Enter);
        assert_eq!(s.picker, Some(1));
        let mut mismatched = first;
        mismatched.id = "lma:some-other-name:latest".into();
        assert_eq!(
            selection_reference(&mismatched),
            PathBuf::from("/models/shared.gguf")
        );
    }

    #[test]
    fn gpu_and_context_cycles_mark_explicit_load_without_duplicates() {
        let mut s = setup();
        assert!(s.config.serve.explicit_load.is_empty());
        s.row = 1;
        key(&mut s, KeyCode::Right);
        key(&mut s, KeyCode::Left);
        assert_eq!(s.config.serve.explicit_load, ["num_gpu"]);
        s.row = 2;
        key(&mut s, KeyCode::Right);
        key(&mut s, KeyCode::Left);
        assert_eq!(s.config.serve.explicit_load, ["num_gpu", "num_ctx"]);
    }

    #[test]
    fn numeric_load_edits_are_explicit_only_after_successful_commit() {
        let mut s = setup();
        s.row = 1;
        key(&mut s, KeyCode::Enter);
        s.edit = Some(Editor::new("invalid".into()));
        key(&mut s, KeyCode::Enter);
        assert!(s.config.serve.explicit_load.is_empty());
        key(&mut s, KeyCode::Esc);
        key(&mut s, KeyCode::Enter);
        key(&mut s, KeyCode::Esc);
        assert!(s.config.serve.explicit_load.is_empty());
        key(&mut s, KeyCode::Enter);
        s.edit = Some(Editor::new("0".into()));
        key(&mut s, KeyCode::Enter);
        assert_eq!(s.config.serve.gpu_layers, 0);
        assert_eq!(s.config.serve.explicit_load, ["num_gpu"]);
        s.row = 2;
        key(&mut s, KeyCode::Enter);
        s.edit = Some(Editor::new("8192".into()));
        key(&mut s, KeyCode::Enter);
        assert_eq!(s.config.serve.context, 8192);
        assert_eq!(s.config.serve.explicit_load, ["num_gpu", "num_ctx"]);
    }

    #[test]
    fn chat_has_only_model_gpu_context_start_and_never_changes_mode() {
        let mut s = setup();
        assert_eq!(s.rows().len(), 4);
        let text = rendered(&s);
        for label in ["Model     ", "GPU       ", "Context   ", "Start     "] {
            assert!(text.contains(label));
        }
        for label in ["Mode      ", "Target    ", "API       ", "Client    "] {
            assert!(!text.contains(label));
        }
        s.row = 2;
        key(&mut s, KeyCode::Enter);
        s.edit = Some(Editor::new("8192".into()));
        key(&mut s, KeyCode::Enter);
        assert_eq!(s.config.serve.context, 8192);
        s.row = 3;
        assert!(key(&mut s, KeyCode::Enter).is_none());
        assert!(s.notice.contains("-p FILE"));
        assert_eq!(s.config.mode, "chat");
    }
    #[test]
    fn launch_cycles_all_targets_between_model_and_gpu_without_changing_presets() {
        let mut s = setup();
        s.config.mode = "launch".into();
        s.config.serve.model = "/explicit/model.gguf".into();
        assert_eq!(s.rows().len(), 5);
        let text = rendered(&s);
        let model = text.find("Model     ").unwrap();
        let target = text.find("Target    ").unwrap();
        let gpu = text.find("GPU       ").unwrap();
        let context = text.find("Context   ").unwrap();
        assert!(model < target && target < gpu && gpu < context);
        assert!(!text.contains("Client    "));
        assert!(!text.contains("API       "));
        key(&mut s, KeyCode::Down);
        assert_eq!(s.row, 1);
        for expected in LaunchTarget::ALL
            .into_iter()
            .skip(1)
            .chain([LaunchTarget::Chat])
        {
            key(&mut s, KeyCode::Right);
            assert_eq!(s.config.target, expected);
            assert!(rendered(&s).contains(expected.label()));
            assert!(s.edit.is_none());
            assert!(s.picker.is_none());
            assert_eq!(s.config.serve.client, "default");
            assert_eq!(s.config.serve.api, "all");
            assert_eq!(s.config.mode, "launch");
        }
        key(&mut s, KeyCode::Left);
        assert_eq!(s.config.target, LaunchTarget::ClaudeDesktop);
        key(&mut s, KeyCode::Enter);
        assert_eq!(s.config.target, LaunchTarget::Chat);
        key(&mut s, KeyCode::Char(' '));
        assert_eq!(s.config.target, LaunchTarget::Opencode);
        s.row = 3;
        key(&mut s, KeyCode::Enter);
        s.edit = Some(Editor::new("8192".into()));
        key(&mut s, KeyCode::Enter);
        assert_eq!(s.config.serve.context, 8192);
        s.row = 4;
        assert_eq!(key(&mut s, KeyCode::Enter), Some(true));
        assert_eq!(s.config.target, LaunchTarget::Opencode);
    }

    #[test]
    fn launch_requires_a_model_and_renders_safely_on_tiny_terminals() {
        let mut s = setup();
        s.config.mode = "launch".into();
        s.config.target = LaunchTarget::CodexDesktop;
        s.row = 4;
        assert!(key(&mut s, KeyCode::Enter).is_none());
        assert!(s.notice.contains("-p FILE"));
        for (width, height) in [(1, 1), (10, 5), (40, 12)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|frame| draw(frame, &s)).unwrap();
        }
    }

    #[test]
    fn hosting_keeps_api_and_client_options_but_no_mode_row() {
        let mut s = setup();
        s.config.mode = "host".into();
        assert_eq!(s.rows().len(), 6);
        let text = rendered(&s);
        assert!(text.contains("API       "));
        assert!(text.contains("Client    "));
        assert!(!text.contains("Mode      "));
        assert!(!text.contains("Target    "));
        s.row = 4;
        key(&mut s, KeyCode::Right);
        assert_eq!(s.config.serve.client, "claude-code");
        key(&mut s, KeyCode::Right);
        assert_eq!(s.config.serve.client, "codex");
    }
    #[test]
    fn explicit_path_override_is_preserved_until_a_model_is_chosen() {
        let mut config = setup().config;
        config.serve.model = "/explicit/model.gguf".into();
        let mut s = Setup::new(
            config,
            ModelCatalog {
                models: vec![model("installed", "lma")],
                warnings: vec![],
            },
        );
        assert!(rendered(&s).contains("Path override: /explicit/model.gguf"));
        key(&mut s, KeyCode::Enter);
        key(&mut s, KeyCode::Esc);
        assert_eq!(s.config.serve.model, PathBuf::from("/explicit/model.gguf"));
        s.row = 3;
        assert_eq!(key(&mut s, KeyCode::Enter), Some(true));
    }
    #[test]
    fn empty_picker_is_safe_and_pasting_never_changes_the_model() {
        let mut s = setup();
        key(&mut s, KeyCode::Enter);
        for code in [
            KeyCode::Down,
            KeyCode::Up,
            KeyCode::Home,
            KeyCode::End,
            KeyCode::Enter,
            KeyCode::Char('/'),
        ] {
            key(&mut s, code);
        }
        assert!(s.edit.is_none());
        assert!(s.config.serve.model.as_os_str().is_empty());
        assert!(rendered(&s).contains("No installed models found"));
        assert!(rendered(&s).contains("-p FILE"));
        key(&mut s, KeyCode::Esc);
        assert_eq!(key(&mut s, KeyCode::Esc), Some(false));
    }
    #[test]
    fn picker_scrolls_and_tiny_terminals_do_not_panic() {
        let mut s = Setup::new(
            setup().config,
            ModelCatalog {
                models: (0..40)
                    .map(|n| model(&format!("model-{n}"), "lma"))
                    .collect(),
                warnings: vec![],
            },
        );
        key(&mut s, KeyCode::Enter);
        key(&mut s, KeyCode::End);
        assert!(rendered(&s).contains("[lma] model-39"));
        key(&mut s, KeyCode::PageUp);
        assert_eq!(s.picker, Some(29));
        for (w, h) in [(1, 1), (10, 5), (40, 12)] {
            let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
            terminal.draw(|f| draw(f, &s)).unwrap();
        }
    }
}
