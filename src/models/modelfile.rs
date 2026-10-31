//! Bounded, text-only Ollama-style model definitions.
//!
//! Supported directives are FROM, SYSTEM, TEMPLATE, LICENSE, MESSAGE and the
//! scalar PARAMETER subset below. TEMPLATE selects an existing lma built-in;
//! arbitrary Ollama Go templates, adapters, training and execution are rejected.
//! Quoted scalars support escaping the quote/backslash; triple-quoted text is
//! literal, including newlines and hashes. Parsing never resolves or downloads
//! weights. Preparation resolves only the explicit FROM through a caller-supplied
//! installed/local-model resolver; relative files are based on the Modelfile.
use serde::{Deserialize, Serialize};
use std::{
    io::{Read, Write},
    path::Path,
};

use super::registry;

pub const MAX_BYTES: usize = 1024 * 1024;
pub const MAX_MESSAGES: usize = 4096;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Definition {
    pub from: String,
    pub system: Option<String>,
    pub template: Option<String>,
    pub license: Option<String>,
    pub messages: Vec<DefinitionMessage>,
    pub parameters: Parameters,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DefinitionMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Parameters {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub seed: Option<u32>,
    pub num_predict: Option<u32>,
    pub num_ctx: Option<u32>,
    pub num_gpu: Option<u32>,
    pub num_thread: Option<i32>,
    pub num_batch: Option<u32>,
    pub u_batch: Option<u32>,
}

fn text(value: &str) -> Result<(), String> {
    if value.contains('\0') {
        return Err("text must not contain NUL bytes".into());
    }
    Ok(())
}

fn template(value: &str) -> Result<(), String> {
    if !crate::config::CHAT_TEMPLATES.contains(&value) {
        return Err(format!(
            "custom Ollama Go TEMPLATE text is unsupported; select an lma built-in template: {}",
            crate::config::CHAT_TEMPLATES.join(", ")
        ));
    }
    Ok(())
}

impl DefinitionMessage {
    pub fn validate(&self) -> Result<(), String> {
        if !matches!(self.role.as_str(), "system" | "user" | "assistant") {
            return Err("MESSAGE role must be system, user, or assistant (text only)".into());
        }
        text(&self.content)
    }
}

impl Parameters {
    pub fn validate(&self) -> Result<(), String> {
        if self
            .temperature
            .is_some_and(|v| !v.is_finite() || !(0.0..=2.0).contains(&v))
        {
            return Err("temperature must be finite and between 0 and 2".into());
        }
        if self
            .top_p
            .is_some_and(|v| !v.is_finite() || v <= 0.0 || v > 1.0)
        {
            return Err("top_p must be finite, greater than 0, and at most 1".into());
        }
        if self.num_predict == Some(0) {
            return Err("num_predict must be a positive u32".into());
        }
        if self
            .num_ctx
            .is_some_and(|v| !(2..=i32::MAX as u32).contains(&v))
        {
            return Err("num_ctx must be between 2 and i32::MAX".into());
        }
        if self.num_thread.is_some_and(|v| v <= 0) {
            return Err("num_thread must be a positive i32".into());
        }
        for (name, value) in [("num_batch", self.num_batch), ("u_batch", self.u_batch)] {
            if value.is_some_and(|v| v == 0 || v > i32::MAX as u32) {
                return Err(format!("{name} must be positive and at most i32::MAX"));
            }
        }
        if let (Some(batch), Some(micro)) = (self.num_batch, self.u_batch)
            && micro > batch
        {
            return Err("u_batch must not exceed num_batch".into());
        }
        Ok(())
    }

    fn set(&mut self, name: &str, value: &str) -> Result<(), String> {
        fn number<T: std::str::FromStr>(name: &str, value: &str) -> Result<T, String> {
            value
                .parse()
                .map_err(|_| format!("invalid {name} value '{value}'"))
        }
        // Validate each occurrence, but check the cross-field batch constraint
        // only after all last-wins scalar assignments have been applied.
        let mut scalar = Self::default();
        match name {
            "temperature" | "top_p" => {
                // Check the input before narrowing to the engine's f32 so values
                // just outside a bound cannot round back into the allowed range.
                let parsed: f64 = number(name, value)?;
                let valid = parsed.is_finite()
                    && if name == "temperature" {
                        (0.0..=2.0).contains(&parsed)
                    } else {
                        parsed > 0.0 && parsed <= 1.0
                    };
                if !valid {
                    return Err(format!(
                        "invalid {name} value '{value}': outside the supported range"
                    ));
                }
                if name == "temperature" {
                    scalar.temperature = Some(parsed as f32);
                } else {
                    scalar.top_p = Some(parsed as f32);
                }
            }
            "seed" => scalar.seed = Some(number(name, value)?),
            "num_predict" => scalar.num_predict = Some(number(name, value)?),
            "num_ctx" => scalar.num_ctx = Some(number(name, value)?),
            "num_gpu" => scalar.num_gpu = Some(number(name, value)?),
            "num_thread" => scalar.num_thread = Some(number(name, value)?),
            "num_batch" => scalar.num_batch = Some(number(name, value)?),
            "u_batch" => scalar.u_batch = Some(number(name, value)?),
            _ => {
                return Err(format!(
                    "unsupported PARAMETER '{name}'; supported: temperature, top_p, seed, num_predict, num_ctx, num_gpu, num_thread, num_batch, u_batch"
                ));
            }
        }
        scalar.validate()?;
        match name {
            "temperature" => self.temperature = scalar.temperature,
            "top_p" => self.top_p = scalar.top_p,
            "seed" => self.seed = scalar.seed,
            "num_predict" => self.num_predict = scalar.num_predict,
            "num_ctx" => self.num_ctx = scalar.num_ctx,
            "num_gpu" => self.num_gpu = scalar.num_gpu,
            "num_thread" => self.num_thread = scalar.num_thread,
            "num_batch" => self.num_batch = scalar.num_batch,
            "u_batch" => self.u_batch = scalar.u_batch,
            _ => unreachable!(),
        }
        Ok(())
    }
}

impl Definition {
    /// Validate deserialized or programmatic definitions before use/publication.
    pub fn validate(&self) -> Result<(), String> {
        if self.from.trim().is_empty() || self.from.contains(['\n', '\r']) {
            return Err("FROM must be one nonempty installed name or local file path".into());
        }
        if self.from.contains("://") {
            return Err("FROM accepts installed names or local files, not remote URLs; install/import the model first".into());
        }
        text(&self.from)?;
        for value in [&self.system, &self.license].into_iter().flatten() {
            text(value)?;
        }
        if let Some(value) = &self.template {
            template(value)?;
        }
        if self.messages.len() > MAX_MESSAGES {
            return Err(format!("definition exceeds {MAX_MESSAGES} MESSAGE records"));
        }
        for message in &self.messages {
            message.validate()?;
        }
        self.parameters.validate()?;
        // Count serialized bytes without allocating a second, possibly huge copy.
        struct Budget(usize);
        impl Write for Budget {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0 = self
                    .0
                    .checked_add(bytes.len())
                    .filter(|n| *n <= MAX_BYTES)
                    .ok_or_else(|| std::io::Error::other("definition exceeds 1 MiB"))?;
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        serde_json::to_writer(Budget(0), self).map_err(|error| error.to_string())
    }
}

fn error(line: usize, message: impl std::fmt::Display) -> String {
    format!("Modelfile line {line}: {message}")
}

fn word(value: &str) -> (&str, &str) {
    let value = value.trim_start();
    match value.find(char::is_whitespace) {
        Some(end) => (&value[..end], value[end..].trim_start()),
        None => (value, ""),
    }
}

fn uncomment(value: &str) -> Result<&str, String> {
    let mut quote = None;
    let mut escaped = false;
    for (index, ch) in value.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if quote.is_some() && ch == '\\' {
            escaped = true;
            continue;
        }
        if let Some(open) = quote {
            if ch == open {
                quote = None;
            }
        } else if matches!(ch, '\'' | '"') {
            quote = Some(ch);
        } else if ch == '#' {
            return Ok(value[..index].trim_end());
        }
    }
    if quote.is_some() {
        return Err("unterminated quoted value".into());
    }
    Ok(value.trim_end())
}

fn scalar(value: &str) -> Result<String, String> {
    let value = uncomment(value)?.trim();
    if value.is_empty() {
        return Err("expected a value (use quotes for empty text)".into());
    }
    if value.starts_with("\"\"\"") {
        return Err(
            "triple quotes are supported only for SYSTEM, TEMPLATE, LICENSE, and MESSAGE content"
                .into(),
        );
    }
    if let Some(quote) = value.chars().next().filter(|c| matches!(c, '\'' | '"')) {
        let mut output = String::new();
        let mut chars = value[quote.len_utf8()..].char_indices();
        while let Some((index, ch)) = chars.next() {
            if ch == quote {
                let tail = &value[quote.len_utf8() + index + ch.len_utf8()..];
                if !tail.trim().is_empty() {
                    return Err("unexpected text after quoted value".into());
                }
                return Ok(output);
            }
            if ch == '\\' {
                let (_, next) = chars.next().ok_or("unterminated escape in quoted value")?;
                if next != quote && next != '\\' {
                    output.push('\\');
                }
                output.push(next);
            } else {
                output.push(ch);
            }
        }
        return Err("unterminated quoted value".into());
    }
    Ok(value.into())
}

struct Lines<'a> {
    source: &'a str,
    offset: usize,
    line: usize,
}
impl<'a> Lines<'a> {
    fn next(&mut self) -> Option<(usize, &'a str)> {
        if self.offset == self.source.len() {
            return None;
        }
        let rest = &self.source[self.offset..];
        let end = rest.find('\n').unwrap_or(rest.len());
        self.offset += end + usize::from(end < rest.len());
        self.line += 1;
        Some((self.line, &rest[..end]))
    }

    fn value(&mut self, value: &str, line: usize) -> Result<String, String> {
        let Some(mut rest) = value.strip_prefix("\"\"\"") else {
            return scalar(value).map_err(|e| error(line, e));
        };
        let mut output = String::new();
        loop {
            if let Some(end) = rest.find("\"\"\"") {
                output.push_str(&rest[..end]);
                let tail = rest[end + 3..].trim_start();
                if !tail.is_empty() && !tail.starts_with('#') {
                    return Err(error(
                        self.line,
                        "unexpected text after closing triple quotes",
                    ));
                }
                return Ok(output);
            }
            output.push_str(rest);
            let Some((_, next)) = self.next() else {
                return Err(error(line, "unterminated triple-quoted value"));
            };
            output.push('\n');
            rest = next;
        }
    }
}

pub fn parse(source: &str) -> Result<Definition, String> {
    if source.len() > MAX_BYTES {
        return Err(error(1, "Modelfile exceeds 1 MiB"));
    }
    if source.contains('\0') {
        return Err(error(
            source[..source.find('\0').unwrap()]
                .bytes()
                .filter(|b| *b == b'\n')
                .count()
                + 1,
            "NUL bytes are not allowed",
        ));
    }
    let mut lines = Lines {
        source,
        offset: 0,
        line: 0,
    };
    let mut definition = Definition::default();
    let mut from_seen = false;
    let mut last_batch_line = 1;
    while let Some((line, raw)) = lines.next() {
        let raw = raw.trim_start();
        if raw.is_empty() || raw.starts_with('#') {
            continue;
        }
        let (directive, args) = word(raw);
        match directive.to_ascii_uppercase().as_str() {
            "FROM" => {
                if from_seen {
                    return Err(error(line, "FROM must occur exactly once"));
                }
                definition.from = scalar(args).map_err(|e| error(line, e))?;
                let candidate = Definition {
                    from: definition.from.clone(),
                    ..Default::default()
                };
                candidate.validate().map_err(|e| error(line, e))?;
                from_seen = true;
            }
            "SYSTEM" => definition.system = Some(lines.value(args, line)?),
            "LICENSE" => definition.license = Some(lines.value(args, line)?),
            "TEMPLATE" => {
                let value = lines.value(args, line)?.trim().to_owned();
                template(&value).map_err(|e| error(line, e))?;
                definition.template = Some(value);
            }
            "MESSAGE" => {
                let (role, content) = word(args);
                if !matches!(role, "system" | "user" | "assistant") {
                    return Err(error(
                        line,
                        "MESSAGE role must be system, user, or assistant (text only)",
                    ));
                }
                if definition.messages.len() == MAX_MESSAGES {
                    return Err(error(
                        line,
                        format!("definition exceeds {MAX_MESSAGES} MESSAGE records"),
                    ));
                }
                definition.messages.push(DefinitionMessage {
                    role: role.into(),
                    content: lines.value(content, line)?,
                });
            }
            "PARAMETER" => {
                let (name, value) = word(args);
                let value = scalar(value).map_err(|e| error(line, e))?;
                definition
                    .parameters
                    .set(name, &value)
                    .map_err(|e| error(line, e))?;
                if matches!(name, "num_batch" | "u_batch") {
                    last_batch_line = line;
                }
            }
            _ => {
                return Err(error(
                    line,
                    format!(
                        "unsupported directive '{directive}'; supported: FROM, SYSTEM, TEMPLATE (lma built-ins only), LICENSE, MESSAGE, PARAMETER; adapters/training are not supported"
                    ),
                ));
            }
        }
    }
    if !from_seen {
        return Err(error(1, "FROM is required exactly once"));
    }
    definition
        .parameters
        .validate()
        .map_err(|e| error(last_batch_line, e))?;
    definition
        .validate()
        .map_err(|e| error(lines.line.max(1), e))?;
    Ok(definition)
}

pub fn read(path: &Path) -> Result<Definition, String> {
    let mut bytes = Vec::new();
    crate::local_file::open_regular(path)?
        .take(MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > MAX_BYTES {
        return Err(error(1, "Modelfile exceeds 1 MiB"));
    }
    let source = std::str::from_utf8(&bytes).map_err(|invalid| {
        let line = bytes[..invalid.valid_up_to()]
            .iter()
            .filter(|b| **b == b'\n')
            .count()
            + 1;
        error(line, "Modelfile must be UTF-8")
    })?;
    parse(source)
}

/// Resolve a definition without publishing it. The resolver must only consult
/// installed/local models; this module does not download, copy, or load weights.
pub fn prepare_definition(
    mut definition: Definition,
    parent: &Path,
    resolve: impl FnOnce(&str) -> Result<registry::Entry, String>,
) -> Result<(Definition, registry::Entry), String> {
    definition.validate()?;
    let from = Path::new(&definition.from);
    let candidate = parent.join(from);
    let local = from.is_absolute()
        || definition.from.starts_with('.')
        || definition.from.contains('\\')
        || from.extension().is_some_and(|e| e == "gguf")
        || candidate.is_file();
    if local {
        let absolute = std::fs::canonicalize(&candidate).map_err(|e| {
            format!(
                "cannot resolve Modelfile FROM '{}': {e}",
                candidate.display()
            )
        })?;
        definition.from = absolute
            .to_str()
            .ok_or("Modelfile FROM path must be UTF-8")?
            .into();
    }
    definition.validate()?;
    let entry = resolve(&definition.from)?;
    Ok((definition, entry))
}

pub fn prepare(
    path: &Path,
    resolve: impl FnOnce(&str) -> Result<registry::Entry, String>,
) -> Result<(Definition, registry::Entry), String> {
    prepare_definition(
        read(path)?,
        path.parent().unwrap_or(Path::new(".")),
        resolve,
    )
}

/// Atomically publish the resolved alias and definition in one registry update.
pub fn create(
    root: &Path,
    label: &str,
    path: &Path,
    resolve: impl FnOnce(&str) -> Result<registry::Entry, String>,
) -> Result<String, String> {
    let (definition, entry) = prepare(path, resolve)?;
    registry::put_definition(root, label, entry, definition)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_literal_multiline_comments_history_and_last_wins_scalars() {
        let source = "# header\nFROM \"./weights #1.gguf\" # source\nSYSTEM old\nSYSTEM \"\"\"first # literal\nsecond \"quoted\" # still literal\"\"\" # tail\nLICENSE \"\"\"license\n# retained\"\"\"\nTEMPLATE \"\"\"chatml\"\"\"\nMESSAGE user \"hello # world\" # tail\nMESSAGE assistant \"\"\"reply\n# literal\"\"\"\nPARAMETER temperature 1\nPARAMETER temperature 0\nPARAMETER seed 4294967295\nPARAMETER num_gpu 0\nPARAMETER u_batch 512\nPARAMETER num_batch 1024\n";
        let result = parse(source).unwrap();
        assert_eq!(result.from, "./weights #1.gguf");
        assert_eq!(
            result.system.as_deref(),
            Some("first # literal\nsecond \"quoted\" # still literal")
        );
        assert_eq!(result.license.as_deref(), Some("license\n# retained"));
        assert_eq!(result.template.as_deref(), Some("chatml"));
        assert_eq!(result.messages[0].content, "hello # world");
        assert_eq!(result.messages[1].content, "reply\n# literal");
        assert_eq!(result.parameters.temperature, Some(0.0));
        assert_eq!(result.parameters.seed, Some(u32::MAX));
        assert_eq!(result.parameters.num_gpu, Some(0));
        result.validate().unwrap();
    }

    #[test]
    fn every_supported_parameter_is_typed_bounded_and_last_wins() {
        let valid = "FROM model:latest\nPARAMETER temperature 2\nPARAMETER top_p 1\nPARAMETER seed 0\nPARAMETER num_predict 4294967295\nPARAMETER num_ctx 2147483647\nPARAMETER num_gpu 4294967295\nPARAMETER num_thread 2147483647\nPARAMETER num_batch 2147483647\nPARAMETER u_batch 2147483647\nPARAMETER num_predict 1\n";
        let result = parse(valid).unwrap();
        assert_eq!(result.parameters.num_predict, Some(1));
        for parameter in [
            "temperature NaN",
            "temperature inf",
            "temperature -1",
            "temperature 2.1",
            "temperature 2.000000001",
            "top_p 1.000000001",
            "top_p 0",
            "top_p 1.1",
            "seed -1",
            "seed 4294967296",
            "num_predict 0",
            "num_ctx 1",
            "num_ctx 2147483648",
            "num_gpu -1",
            "num_thread 0",
            "num_thread 2147483648",
            "num_batch 0",
            "u_batch 2147483648",
            "stop x",
            "num_ubatch 1",
        ] {
            let error = parse(&format!("FROM model\nPARAMETER {parameter}")).unwrap_err();
            assert!(error.contains("line 2"), "{parameter}: {error}");
        }
        assert!(parse("FROM model\nPARAMETER num_batch 4\nPARAMETER u_batch 5").is_err());
        assert!(parse("FROM model\nPARAMETER temperature 3\nPARAMETER temperature 1").is_err());
    }

    #[test]
    fn errors_have_lines_and_unsupported_meaning_is_never_ignored() {
        for (source, line) in [
            ("SYSTEM hi", 1),
            ("FROM a\nFROM b", 2),
            ("FROM a\nADAPTER ./adapter", 2),
            ("FROM a\nTRAIN yes", 2),
            ("FROM a\nMESSAGE tool text", 2),
            ("FROM a\nSYSTEM \"unfinished", 2),
            ("FROM a\nSYSTEM \"\"\"unfinished\n# text", 2),
            ("FROM a\nSYSTEM \"\"\"ok\"\"\" junk", 2),
            ("FROM a\nTEMPLATE \"\"\"{{ .Prompt }}\"\"\"", 2),
            ("FROM \"\"", 1),
            ("FROM https://example.com/model.gguf", 1),
            ("FROM a\nPARAMETER top_p 1 extra", 2),
        ] {
            let error = parse(source).unwrap_err();
            assert!(error.contains(&format!("line {line}:")), "{error}");
        }
        assert!(
            parse("FROM a\nTEMPLATE {{ .Prompt }}")
                .unwrap_err()
                .contains("built-in")
        );
    }

    #[test]
    fn quote_escapes_hashes_crlf_and_empty_text_are_preserved() {
        let result = parse("FROM a\r\nSYSTEM \"say \\\"#hi\\\"\" # comment\r\nLICENSE ''\r\nMESSAGE system \"\"\"\r\n#hash\r\n\"\"\"\r\n").unwrap();
        assert_eq!(result.system.as_deref(), Some("say \"#hi\""));
        assert_eq!(result.license.as_deref(), Some(""));
        assert_eq!(result.messages[0].content, "\r\n#hash\r\n");
    }

    #[test]
    fn public_validation_and_deserialization_are_strict_and_bounded() {
        assert!(Definition::default().validate().is_err());
        for value in [
            serde_json::json!({"from":"a","unknown":1}),
            serde_json::json!({"from":"a","parameters":{"stop":"x"}}),
            serde_json::json!({"from":"a","messages":[{"role":"user","content":"x","images":[]}]}),
        ] {
            assert!(serde_json::from_value::<Definition>(value).is_err());
        }
        let mut definition = parse("FROM a").unwrap();
        definition.parameters.temperature = Some(f32::NAN);
        assert!(definition.validate().is_err());
        definition.parameters = Parameters::default();
        definition.system = Some("x".repeat(MAX_BYTES));
        assert!(definition.validate().is_err());
        assert!(parse(&"#".repeat(MAX_BYTES + 1)).is_err());
        definition.system = None;
        definition.messages = vec![
            DefinitionMessage {
                role: "user".into(),
                content: String::new()
            };
            MAX_MESSAGES + 1
        ];
        assert!(definition.validate().is_err());
    }

    #[test]
    fn relative_from_is_based_on_modelfile_and_publication_is_atomic() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("weights #1.gguf");
        std::fs::write(&model, b"GGUFfixture").unwrap();
        let file = dir.path().join("Modelfile");
        std::fs::write(
            &file,
            "FROM \"./weights #1.gguf\"\nSYSTEM local\nPARAMETER num_ctx 128",
        )
        .unwrap();
        let root = dir.path().join("registry");
        let label = create(&root, "local", &file, |from| {
            assert_eq!(Path::new(from), std::fs::canonicalize(&model).unwrap());
            Ok(registry::Entry {
                path: from.into(),
                projector: None,
            })
        })
        .unwrap();
        assert_eq!(label, "local:latest");
        assert_eq!(
            registry::definition(&root, &label)
                .unwrap()
                .unwrap()
                .system
                .as_deref(),
            Some("local")
        );
        let before = std::fs::read(root.join(".registry-v1.json")).unwrap();
        assert!(
            create(&root, "local", &file, |from| Ok(registry::Entry {
                path: from.into(),
                projector: None
            }))
            .is_err()
        );
        assert_eq!(
            std::fs::read(root.join(".registry-v1.json")).unwrap(),
            before
        );
        assert!(create(&root, "failed", &file, |_| Err("resolver failed".into())).is_err());
        assert!(!registry::read(&root).unwrap().contains_key("failed:latest"));
        assert_eq!(std::fs::read(&model).unwrap(), b"GGUFfixture");
    }

    #[test]
    fn file_reads_are_bounded_and_invalid_utf8_reports_its_line() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("Modelfile");
        std::fs::write(&file, b"FROM model\nSYSTEM \xff").unwrap();
        assert!(read(&file).unwrap_err().contains("line 2:"));
        std::fs::write(&file, vec![b'#'; MAX_BYTES + 1]).unwrap();
        assert!(read(&file).unwrap_err().contains("1 MiB"));
    }

    #[test]
    fn installed_names_are_passed_unchanged_to_only_the_injected_resolver() {
        let dir = tempfile::tempdir().unwrap();
        let definition = parse("FROM team/model:tag").unwrap();
        let result = prepare_definition(definition, dir.path(), |from| {
            assert_eq!(from, "team/model:tag");
            Err("not installed; no downloads".into())
        });
        assert!(result.unwrap_err().contains("not installed"));
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }
}
