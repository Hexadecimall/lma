use super::{
    attachments::{self, Attachment},
    editor::Editor,
    logs, markdown,
    text::{one_line, sanitize},
};
use crate::config::ServeConfig;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEventKind};
use serde_json::{Value, json};
use std::{fs::OpenOptions, io::Write, path::Path, time::Instant};

pub const HELP: &str = "Keys\nEnter / Ctrl+Enter send · Shift+Enter newline · Alt+Enter fallback\nEnhanced keyboard input is enabled on supporting terminals; legacy terminals may collapse Shift+Enter into Enter.\n←/→ Home/End edit · ↑/↓ move between input lines · Ctrl+A/E line start/end · Ctrl+U clear to line start\nMouse wheel / PgUp/PgDn scroll · Tab switch composer/history · history ↑/↓ scroll\nCtrl+Home/End oldest/latest from either focus · history Home/End oldest/latest\nScrolled history stays in place as replies stream; Ctrl+End resumes following output.\nEsc cancel generation / close panel · Ctrl+C cancel while busy, quit while idle · Ctrl+Q quit\n\nCommands\n/help                 this panel; Esc closes\n/new                  clear in-memory conversation and attachments (keep system/sampling)\n/models               return to setup; unload this model and clear conversation\n/system TEXT          replace system prompt; /system alone clears it\n/tokens N|off         reply cap; off = no reply cap, bounded by remaining context\n/tokens               show current reply cap; model remains loaded at the limit\n/temperature N        sampling temperature, 0–2 (0 = greedy)\n/raw                  raw completion: composer only, no history/system/template/images\n/chat                 native chat template with history and optional system\n/markdown             render assistant Markdown (default)\n/plain                show literal assistant text, without Markdown formatting\n/image FILE           attach PNG/JPEG; requires a vision projector; path is literal, no shell expansion\n/save FILE            export completed text conversation; new file only, never overwrite\n/logs                 latest bounded native logs (512 entries, 4 KiB each); Esc closes\n/quit                 quit; nothing is saved automatically\n\nMarkdown: headings, emphasis, code, lists, quotes and visible links. Code/links never execute.\nNo tools, code execution, external agents, or hidden history. /plain shows literal Markdown.\nImages: at most 8 per request, 8 MiB each, 16 MiB decoded total, 24 MiB JSON (history included).\nThe engine checks exact prompt + output context budget; history is never silently trimmed.\nReplies stream as they are generated. Cancelled or failed partial replies are not added to history.";

#[derive(Clone)]
pub struct Message {
    pub role: &'static str,
    pub text: String,
    pub images: Vec<Attachment>,
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    Composer,
    History,
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Panel {
    Help,
    Logs,
}
#[derive(Debug, PartialEq, Eq)]
pub enum Action {
    None,
    Send,
    Cancel,
    Quit,
    Models,
}

pub struct App {
    pub config: ServeConfig,
    pub model: String,
    pub vision: bool,
    pub editor: Editor,
    pub images: Vec<Attachment>,
    pub history: Vec<Message>,
    pub system: String,
    pub raw: bool,
    pub tokens: u32,
    pub temperature: f32,
    pub busy: Option<Instant>,
    pub focus: Focus,
    pub panel: Option<Panel>,
    /// Distance from the bottom, in wrapped display lines.
    pub scroll: usize,
    history_max_scroll: usize,
    pub history_height: usize,
    pub markdown_enabled: bool,
    pub markdown_cache: markdown::Cache,
    pub notice: String,
    pub usage: String,
    pub partial: String,
    pub partial_status: Option<&'static str>,
}

impl App {
    pub fn new(config: ServeConfig, vision: bool) -> Self {
        let model = one_line(
            &config
                .model_id
                .clone()
                .unwrap_or_else(|| config.model.to_string_lossy().into_owned()),
        );
        let tokens = config
            .chat_tokens
            .unwrap_or_else(|| 256.min(config.context.saturating_sub(1)).max(1));
        Self {
            config,
            model,
            vision,
            editor: Editor::default(),
            images: vec![],
            history: vec![],
            system: String::new(),
            raw: false,
            tokens,
            temperature: 0.8,
            busy: None,
            focus: Focus::Composer,
            panel: None,
            scroll: 0,
            history_max_scroll: 0,
            history_height: 10,
            markdown_enabled: true,
            markdown_cache: markdown::Cache::default(),
            notice: "Local, memory-only chat · /help for keys and commands".into(),
            usage: String::new(),
            partial: String::new(),
            partial_status: None,
        }
    }
    /// Keep the same top visible line while content grows, unless following live output.
    pub fn viewport(&mut self, lines: usize, height: usize) -> usize {
        let max = lines.saturating_sub(height);
        if self.scroll > 0 && self.scroll != usize::MAX {
            let top = self.history_max_scroll.saturating_sub(self.scroll);
            self.scroll = max.saturating_sub(top);
        }
        self.history_max_scroll = max;
        self.history_height = height;
        self.scroll = self.scroll.min(max);
        max.saturating_sub(self.scroll)
    }

    pub fn mouse(&mut self, event: MouseEventKind) {
        match event {
            MouseEventKind::ScrollUp => self.scroll = self.scroll.saturating_add(3),
            MouseEventKind::ScrollDown => self.scroll = self.scroll.saturating_sub(3),
            _ => {}
        }
    }

    pub fn key(&mut self, key: KeyEvent) -> Action {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('q') if ctrl => return Action::Quit,
            KeyCode::Char('c') if ctrl => {
                return if self.busy.is_some() {
                    Action::Cancel
                } else {
                    Action::Quit
                };
            }
            KeyCode::Esc => {
                if self.busy.is_some() {
                    return Action::Cancel;
                }
                self.panel = None;
                self.focus = Focus::Composer;
                self.scroll = 0;
                return Action::None;
            }
            KeyCode::Tab | KeyCode::BackTab => {
                self.focus = if self.focus == Focus::Composer {
                    Focus::History
                } else {
                    Focus::Composer
                };
                return Action::None;
            }
            KeyCode::Home if ctrl => {
                self.scroll = usize::MAX;
                return Action::None;
            }
            KeyCode::End if ctrl => {
                self.scroll = 0;
                return Action::None;
            }
            KeyCode::PageUp => {
                self.scroll = self
                    .scroll
                    .saturating_add(self.history_height.saturating_sub(1).max(1));
                return Action::None;
            }
            KeyCode::PageDown => {
                self.scroll = self
                    .scroll
                    .saturating_sub(self.history_height.saturating_sub(1).max(1));
                return Action::None;
            }
            _ => {}
        }
        if self.focus == Focus::History || self.panel.is_some() {
            match key.code {
                KeyCode::Up => self.scroll = self.scroll.saturating_add(1),
                KeyCode::Down => self.scroll = self.scroll.saturating_sub(1),
                KeyCode::Home => self.scroll = usize::MAX,
                KeyCode::End => self.scroll = 0,
                _ => {}
            }
            return Action::None;
        }
        if self.busy.is_some() {
            return Action::None;
        }
        if key.code == KeyCode::Enter {
            if key
                .modifiers
                .intersects(KeyModifiers::ALT | KeyModifiers::SHIFT)
            {
                self.editor.insert("\n");
                return Action::None;
            }
            if self.editor.text.starts_with('/') {
                return self.command();
            }
            if !self.editor.text.trim().is_empty() || !self.images.is_empty() {
                return Action::Send;
            }
        } else {
            self.editor.key(key);
        }
        Action::None
    }
    fn command(&mut self) -> Action {
        let input = self.editor.text.trim().to_owned();
        let (name, arg) = input
            .split_once(char::is_whitespace)
            .unwrap_or((&input, ""));
        let arg = arg.trim();
        let mut action = Action::None;
        let result: Result<String, String> = match name {
            "/help" if arg.is_empty() => { self.panel = Some(Panel::Help); self.scroll = usize::MAX; Ok("Help · Esc closes".into()) }
            "/logs" if arg.is_empty() => { self.panel = Some(Panel::Logs); self.scroll = 0; Ok("Native logs · bounded tail, including backend diagnostics · Esc closes".into()) }
            "/new" if arg.is_empty() => { self.history.clear(); self.markdown_cache.clear(); self.images.clear(); self.partial.clear(); self.partial_status = None; self.scroll = 0; self.usage.clear(); Ok("New conversation. System prompt and sampling retained.".into()) }
            "/models" if arg.is_empty() => { action = Action::Models; Ok(String::new()) }
            "/quit" if arg.is_empty() => { action = Action::Quit; Ok(String::new()) }
            "/system" => { self.system = sanitize(arg); Ok(if self.system.is_empty() { "System prompt cleared" } else { "Custom system prompt set (chat mode only)" }.into()) }
            "/tokens" if arg.is_empty() => Ok(format!("Reply limit: {}. Use /tokens N or /tokens off (remaining context).", self.token_limit_label())),
            "/tokens" => crate::config::chat_token_limit(arg)
                .and_then(|n| {
                    if n >= self.config.context { return Err(format!("Reply tokens must be smaller than context ({}); use /tokens off for remaining context", self.config.context)); }
                    self.tokens = n;
                    self.config.chat_tokens = Some(n);
                    Ok(format!("Reply limit: {}. Model stays loaded; Esc cancels generation.", self.token_limit_label()))
                }),
            "/temperature" => arg.parse::<f32>().ok().filter(|n| n.is_finite() && (0.0..=2.0).contains(n))
                .map(|n| { self.temperature = n; format!("Temperature: {n}") }).ok_or_else(|| "Use /temperature N with a finite number from 0 to 2".into()),
            "/markdown" if arg.is_empty() => { self.markdown_enabled = true; self.scroll = 0; Ok("Markdown display enabled; code and links are display-only".into()) }
            "/plain" if arg.is_empty() => { self.markdown_enabled = false; self.scroll = 0; Ok("Literal text display enabled; model output is unchanged".into()) }
            "/raw" if arg.is_empty() => { self.raw = true; Ok("Raw completion: only composer text is sent; history/system/images are not sent".into()) }
            "/chat" if arg.is_empty() => { self.raw = false; Ok("Chat template enabled; conversation history and system prompt are sent".into()) }
            "/image" if !arg.is_empty() => {
                let (count, bytes) = self.image_budget();
                if self.raw { Err("Use /chat before attaching images".into()) }
                else { attachments::read(Path::new(arg), self.vision, count, bytes).map(|image| {
                    let name = one_line(&image.name); self.images.push(image); format!("Attached {name}; /new clears all attachments")
                }) }
            }
            "/save" if !arg.is_empty() => self.export(Path::new(arg)).map(|()| format!("Exported completed conversation to {}", one_line(arg))),
            _ => Err("Unknown command or invalid arguments. Use /help. Paths are literal (no quotes or shell expansion).".into()),
        };
        match result {
            Ok(notice) => {
                self.editor = Editor::default();
                self.notice = notice;
            }
            Err(error) => self.notice = error,
        }
        action
    }
    pub fn token_limit_label(&self) -> String {
        if self.tokens == 0 {
            "off · remaining context".into()
        } else {
            format!("{} tokens", self.tokens)
        }
    }

    pub fn image_budget(&self) -> (usize, usize) {
        self.history
            .iter()
            .flat_map(|m| &m.images)
            .chain(&self.images)
            .fold((0, 0), |(n, b), i| (n + 1, b + i.bytes))
    }
    pub fn request(&self) -> Result<(&'static str, Vec<u8>), String> {
        if self.tokens >= self.config.context {
            return Err(
                "Reply limit must be smaller than context; use /tokens N or /tokens off".into(),
            );
        }
        if self.history.iter().map(|m| m.text.len()).sum::<usize>() > 4 * 1024 * 1024 {
            return Err(
                "Conversation text exceeds 4 MiB; /save then /new before continuing".into(),
            );
        }
        let (count, bytes) = self.image_budget();
        if count > attachments::MAX_IMAGES || bytes > attachments::TOTAL_IMAGE_LIMIT {
            return Err("Image request budget exceeded; use /new".into());
        }
        let mut body = json!({ "max_tokens": self.tokens.max(1), "temperature": self.temperature, "stream": true });
        let (path, limit) = if self.raw {
            if !self.images.is_empty() {
                return Err("Raw mode cannot send attachments; use /chat or /new".into());
            }
            body["prompt"] = json!(self.editor.text);
            body["stream_options"] = json!({"include_usage": true});
            ("/v1/completions", 1024 * 1024)
        } else {
            let mut messages = Vec::new();
            if !self.system.is_empty() {
                messages.push(json!({"role": "system", "content": self.system}));
            }
            for message in &self.history {
                messages.push(message_json(message.role, &message.text, &message.images));
            }
            messages.push(message_json("user", &self.editor.text, &self.images));
            body["messages"] = json!(messages);
            body["stream_options"] = json!({"include_usage": true});
            ("/v1/chat/completions", attachments::REQUEST_LIMIT)
        };
        let body = serde_json::to_vec(&body).map_err(|e| e.to_string())?;
        if body.len() > limit {
            return Err(format!(
                "Serialized request exceeds {} MiB; /save then /new",
                limit / (1024 * 1024)
            ));
        }
        Ok((path, body))
    }
    pub fn begin(&mut self) {
        self.partial.clear();
        self.partial_status = None;
        self.usage.clear();
        self.notice.clear();
        self.busy = Some(Instant::now());
        self.scroll = 0;
    }
    pub fn fail(&mut self, error: &str) {
        self.busy = None;
        self.partial_status = Some("failed · not in history");
        self.notice = format!("{} · draft retained", one_line(error));
    }
    pub fn complete(&mut self, reply: String, usage: &Value) {
        self.partial.clear();
        self.partial_status = None;
        self.history.push(Message {
            role: "user",
            text: std::mem::take(&mut self.editor.text),
            images: std::mem::take(&mut self.images),
        });
        self.editor = Editor::default();
        self.history.push(Message {
            role: "assistant",
            text: reply,
            images: vec![],
        });
        self.usage = match (
            usage["prompt_tokens"].as_u64(),
            usage["completion_tokens"].as_u64(),
        ) {
            (Some(input), Some(output)) => format!("{input} in / {output} out tokens"),
            _ => "Completed · token usage unavailable".into(),
        };
        self.notice = match usage["finish_reason"].as_str() {
            Some("length") if self.tokens == 0 => "Context window reached; model stays loaded. /save then /new, or restart with a larger -c.".into(),
            Some("length") => "Reply limit reached; model stays loaded. /tokens off removes the reply cap; /tokens N changes it.".into(),
            Some("stop") => "Model ended its reply (EOS/stop); still loaded and ready.".into(),
            _ => "Ready · /help".into(),
        };
        self.busy = None;
    }
    pub fn cancel(&mut self) {
        self.busy = None;
        self.partial_status = Some("cancelled · not in history");
        self.notice =
            "Cancelled. Draft retained; worker admission stays held until native inference stops."
                .into();
    }
    pub fn panel_text(&self) -> String {
        match self.panel {
            Some(Panel::Help) => HELP.into(),
            Some(Panel::Logs) => logs::snapshot(),
            None => String::new(),
        }
    }
    fn export(&self, path: &Path) -> Result<(), String> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(path).map_err(|e| {
            format!("cannot create export (existing files are never overwritten): {e}")
        })?;
        let mut text = format!("# lma conversation\n\nModel: {}\n\n", self.model);
        if !self.system.is_empty() {
            text.push_str(&format!("## System\n\n{}\n\n", sanitize(&self.system)));
        }
        for message in &self.history {
            text.push_str(&format!(
                "## {}\n\n{}\n\n",
                message.role,
                sanitize(&message.text)
            ));
            for image in &message.images {
                text.push_str(&format!(
                    "Attachment (not embedded): {}\n\n",
                    one_line(&image.name)
                ));
            }
        }
        file.write_all(text.as_bytes())
            .and_then(|()| file.flush())
            .map_err(|e| format!("export failed; partial file may remain: {e}"))
    }
}

fn message_json(role: &str, text: &str, images: &[Attachment]) -> Value {
    json!({"role": role, "content": text, "images": images.iter().map(|i| &i.encoded).collect::<Vec<_>>()})
}

#[cfg(test)]
mod tests {
    use super::*;
    fn app() -> App {
        App::new(ServeConfig::default(), false)
    }
    fn command(a: &mut App, text: &str) -> Action {
        a.editor = Editor::new(text.into());
        a.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
    }
    #[test]
    fn cancel_retains_unicode_draft_and_control_c_quits_only_when_idle() {
        let mut a = app();
        a.editor.insert("你好🙂");
        a.busy = Some(Instant::now());
        assert_eq!(
            a.key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            Action::Cancel
        );
        a.cancel();
        assert_eq!(a.editor.text, "你好🙂");
        assert_eq!(
            a.key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            Action::Quit
        );
    }
    #[test]
    fn prompt_budget_and_sampling_are_validated_without_silent_trimming() {
        let mut a = app();
        let context = a.config.context;
        command(&mut a, &format!("/tokens {context}"));
        assert_eq!(a.tokens, 256);
        command(&mut a, "/temperature NaN");
        assert_eq!(a.temperature, 0.8);
        command(&mut a, "/tokens 64");
        command(&mut a, "/system Be concise");
        a.editor = Editor::new("hi".into());
        let (_, body) = a.request().unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["max_tokens"], 64);
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert_eq!(body["messages"][0]["content"], "Be concise");
        a.tokens = context;
        assert!(a.request().is_err());
    }
    #[test]
    fn raw_sends_only_composer_and_does_not_fake_tools() {
        let mut a = app();
        command(&mut a, "/raw");
        a.system = "secret".into();
        a.editor.insert("abc");
        let (path, body) = a.request().unwrap();
        assert_eq!(path, "/v1/completions");
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["prompt"], "abc");
        assert_eq!(value["stream"], true);
        assert_eq!(value["stream_options"]["include_usage"], true);
        assert!(value.get("messages").is_none());
        command(&mut a, "/run touch /tmp/nope");
        assert!(a.notice.contains("Unknown command"));
    }
    #[test]
    fn enter_and_newline_match_help() {
        let mut a = app();
        a.editor.insert("界");
        a.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT));
        a.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT));
        assert_eq!(a.editor.text, "界\n\n");
        assert_eq!(
            a.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::CONTROL)),
            Action::Send
        );
    }
    #[test]
    fn partial_replies_are_display_only_and_only_success_commits_a_turn() {
        let mut a = app();
        a.editor.insert("Keep my draft");
        a.begin();
        a.partial.push_str("unfinished");
        a.cancel();
        assert!(a.history.is_empty());
        assert_eq!(a.editor.text, "Keep my draft");
        assert_eq!(a.partial, "unfinished");
        assert_eq!(a.partial_status, Some("cancelled · not in history"));
        let (_, body) = a.request().unwrap();
        assert!(!String::from_utf8(body).unwrap().contains("unfinished"));
        let directory = tempfile::tempdir().unwrap();
        let export = directory.path().join("chat.md");
        a.export(&export).unwrap();
        assert!(
            !std::fs::read_to_string(export)
                .unwrap()
                .contains("unfinished")
        );
        a.begin();
        assert!(a.partial.is_empty());
        a.partial.push_str("failed text");
        a.fail("Stream interrupted");
        assert!(a.history.is_empty());
        assert_eq!(a.partial_status, Some("failed · not in history"));
        assert!(a.notice.contains("draft retained"));
        a.begin();
        a.complete(
            "Successful reply".into(),
            &json!({"prompt_tokens":3,"completion_tokens":2}),
        );
        assert_eq!(a.history.len(), 2);
        assert_eq!(a.history[0].text, "Keep my draft");
        assert_eq!(a.history[1].text, "Successful reply");
        assert!(a.editor.text.is_empty());
        assert!(a.partial.is_empty());
        assert_eq!(a.partial_status, None);
        assert_eq!(a.usage, "3 in / 2 out tokens");
    }
    #[test]
    fn scrolling_anchors_streaming_content_and_end_resumes_live_output() {
        let mut a = app();
        assert_eq!(a.viewport(100, 12), 88);
        a.mouse(MouseEventKind::ScrollUp);
        assert_eq!(a.viewport(100, 12), 85);
        assert_eq!(a.viewport(120, 12), 85);
        a.key(KeyEvent::new(KeyCode::PageUp, KeyModifiers::NONE));
        assert_eq!(a.viewport(120, 12), 74);
        a.key(KeyEvent::new(KeyCode::PageDown, KeyModifiers::NONE));
        assert_eq!(a.viewport(120, 12), 85);
        a.key(KeyEvent::new(KeyCode::Home, KeyModifiers::CONTROL));
        assert_eq!(a.viewport(120, 12), 0);
        assert_eq!(a.viewport(140, 12), 0);
        a.key(KeyEvent::new(KeyCode::End, KeyModifiers::CONTROL));
        assert_eq!(a.viewport(140, 12), 128);
        assert_eq!(a.viewport(150, 12), 138);
        a.mouse(MouseEventKind::ScrollUp);
        a.viewport(150, 12);
        a.editor.insert("question");
        a.busy = Some(Instant::now());
        a.complete("reply".into(), &json!({}));
        assert!(
            a.scroll > 0,
            "completion must not snap a reader to the bottom"
        );
        a.viewport(10, 30);
        assert_eq!(a.scroll, 0);
    }
    #[test]
    fn history_focus_scrolls_without_editing_draft_and_newlines_do_not_send() {
        let mut a = app();
        a.editor.insert("draft");
        a.viewport(80, 10);
        a.key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        a.key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(a.scroll, 1);
        assert_eq!(a.editor.text, "draft");
        a.key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE));
        assert_eq!(a.scroll, 0);
        a.key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        for modifier in [KeyModifiers::SHIFT, KeyModifiers::ALT] {
            assert_eq!(a.key(KeyEvent::new(KeyCode::Enter, modifier)), Action::None);
        }
        assert_eq!(a.editor.text, "draft\n\n");
        assert_eq!(
            a.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            Action::Send
        );
    }
    #[test]
    fn unlimited_caps_are_explicit_and_length_finishes_keep_the_conversation_usable() {
        let mut a = App::new(
            ServeConfig {
                chat_tokens: Some(1024),
                ..Default::default()
            },
            false,
        );
        assert_eq!(a.tokens, 1024);
        command(&mut a, "/tokens off");
        assert_eq!(a.tokens, 0);
        assert!(a.token_limit_label().contains("remaining context"));
        a.editor.insert("keep going");
        let (_, body) = a.request().unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            body["max_tokens"], 1,
            "valid placeholder, overridden only by trusted in-process policy"
        );
        a.complete("long answer".into(), &json!({"finish_reason":"length"}));
        assert!(a.notice.contains("Context window reached"));
        assert!(a.notice.contains("stays loaded"));
        assert_eq!(a.history.len(), 2);
        command(&mut a, "/tokens 512");
        assert_eq!(a.tokens, 512);
        a.editor.insert("next");
        a.complete("another answer".into(), &json!({"finish_reason":"length"}));
        assert!(a.notice.contains("Reply limit reached"));
        command(&mut a, "/tokens 0");
        assert_eq!(a.tokens, 0);
        command(&mut a, "/tokens");
        assert!(a.notice.contains("off"));
        a.editor.insert("still usable");
        assert!(a.request().is_ok());
    }
    #[test]
    fn markdown_switch_changes_display_only_not_request_or_saved_text() {
        let mut a = app();
        command(&mut a, "/plain");
        assert!(!a.markdown_enabled);
        command(&mut a, "/markdown");
        assert!(a.markdown_enabled);
        a.editor.insert("**literal prompt**");
        let (_, body) = a.request().unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["messages"][0]["content"], "**literal prompt**");
        a.complete("**literal reply**".into(), &json!({}));
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("chat.md");
        a.export(&path).unwrap();
        assert!(
            std::fs::read_to_string(path)
                .unwrap()
                .contains("**literal reply**")
        );
    }
    #[test]
    fn export_never_overwrites() {
        let a = app();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chat.md");
        a.export(&path).unwrap();
        let before = std::fs::read(&path).unwrap();
        assert!(a.export(&path).is_err());
        assert_eq!(std::fs::read(path).unwrap(), before);
    }
}
