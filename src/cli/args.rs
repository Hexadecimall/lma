use std::{
    collections::VecDeque,
    ffi::{OsStr, OsString},
};

use super::Command;

pub(super) struct Args(VecDeque<OsString>);

impl Args {
    pub fn new(values: impl IntoIterator<Item = OsString>) -> Self {
        Self(values.into_iter().collect())
    }

    pub fn next(&mut self) -> Option<OsString> {
        self.0.pop_front()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn rest(self) -> Vec<OsString> {
        self.0.into_iter().collect()
    }

    pub fn value(&mut self, flag: &Flag) -> Result<OsString, String> {
        let value = match &flag.inline {
            Some(value) => value.clone(),
            None => self
                .next()
                .ok_or_else(|| format!("{} needs a value", flag.name))?,
        };
        if value.is_empty() || value == "--" {
            return Err(format!("{} needs a non-empty value", flag.name));
        }
        if flag.inline.is_none() && value.as_encoded_bytes().starts_with(b"-") {
            return Err(format!(
                "{} needs a value; use {}=VALUE for a value starting with '-'",
                flag.name, flag.name
            ));
        }
        Ok(value)
    }

    pub fn text(&mut self, flag: &Flag) -> Result<String, String> {
        self.value(flag)?
            .into_string()
            .map_err(|_| format!("{} needs a UTF-8 value", flag.name))
    }
}

pub(super) struct Flag {
    pub name: String,
    inline: Option<OsString>,
}

impl Flag {
    pub fn parse(value: &OsStr) -> Option<Self> {
        let bytes = value.as_encoded_bytes();
        if !bytes.starts_with(b"-") {
            return None;
        }
        let (name, inline) = match bytes.iter().position(|byte| *byte == b'=') {
            Some(index) => {
                let name = std::str::from_utf8(&bytes[..index]).ok()?;
                // Splitting immediately after ASCII '=' preserves an encoded-byte boundary.
                let value = unsafe { OsStr::from_encoded_bytes_unchecked(&bytes[index + 1..]) };
                (name, Some(value.to_os_string()))
            }
            None => (value.to_str()?, None),
        };
        Some(Self {
            name: name.into(),
            inline,
        })
    }

    pub fn switch(&self) -> Result<(), String> {
        if self.inline.is_some() {
            Err(format!("{} does not take a value", self.name))
        } else {
            Ok(())
        }
    }
}

// Normalize only the root syntax; NSS deliberately keeps the existing parser surface.
pub(super) fn short_arguments(command: Command, mut args: Args) -> Result<Args, String> {
    let mut normalized = Vec::new();
    while let Some(argument) = args.next() {
        if !argument.as_encoded_bytes().starts_with(b"-") {
            normalized.push(argument);
            continue;
        }
        let invalid = || {
            format!(
                "unsupported root option '{}'; use canonical short flags (see `lma h`), or long options with `lma nss help` and `lma nss COMMAND ...`; bare -- is not supported at root",
                argument.to_string_lossy()
            )
        };
        let flag = Flag::parse(&argument).ok_or_else(invalid)?;
        let (long, takes_value) = match flag.name.as_str() {
            "-h" => ("--help", false),
            "-d" => ("--dry-run", false),
            short => short_flags(command)
                .iter()
                .find(|(name, _, _)| *name == short)
                .map(|(_, long, takes_value)| (*long, *takes_value))
                .ok_or_else(invalid)?,
        };
        if takes_value {
            let value = args.value(&flag)?;
            if value.to_str().is_none() && !value.as_encoded_bytes().starts_with(b"-") {
                normalized.push(long.into());
                normalized.push(value);
            } else {
                // Dash-prefixed values must stay inline for the downstream Args::value check.
                let mut argument = OsString::from(long);
                argument.push("=");
                argument.push(value);
                normalized.push(argument);
            }
        } else {
            flag.switch()?;
            normalized.push(long.into());
        }
    }
    Ok(Args::new(normalized))
}

fn short_flags(command: Command) -> &'static [(&'static str, &'static str, bool)] {
    match command {
        Command::Serve | Command::Chat | Command::Launch => &[
            ("-m", "--model", true),
            ("-p", "--path", true),
            ("-o", "--ollama", true),
            ("-or", "--ollama-dir", true),
            ("-c", "--ctx-size", true),
            ("-t", "--threads", true),
            ("-ts", "--tokens", true),
            ("-g", "--gpu", true),
            ("-b", "--batch-size", true),
            ("-mh", "--ubatch-size", true),
            ("-f", "--flash-attn", true),
            ("-pr", "--projector", true),
            ("-ce", "--chat-template", true),
            ("-pe", "--prompt-cache", false),
            ("-n", "--no-prompt-cache", false),
            ("-k", "--kv-cache", true),
            ("-mk", "--memory-fallback", true),
            ("-e", "--expert-cache-mib", true),
            ("-a", "--api", true),
            ("-ct", "--client", true),
            ("-ht", "--host", true),
            ("-pt", "--port", true),
            ("-ay", "--api-key", true),
            ("-tt", "--target", true),
            ("-i", "--interactive", false),
        ],
        Command::Stop => &[
            ("-m", "--model", true),
            ("-ht", "--host", true),
            ("-p", "--port", true),
            ("-a", "--api-key", true),
        ],
        Command::List => &[
            ("-j", "--json", false),
            ("-dr", "--dir", true),
            ("-o", "--ollama-dir", true),
        ],
        Command::Fetch => &[
            ("-l", "--list", false),
            ("-i", "--index", true),
            ("-f", "--file", true),
            ("-o", "--out", true),
            ("-oe", "--ollama-database", false),
            ("-le", "--lm-studio-database", false),
        ],
        Command::Image => &[
            ("-m", "--model", true),
            ("-p", "--prompt", true),
            ("-n", "--negative-prompt", true),
            ("-o", "--out", true),
            ("-t", "--tokenizer", true),
            ("-w", "--width", true),
            ("-ht", "--height", true),
            ("-s", "--steps", true),
            ("-sd", "--seed", true),
            ("-g", "--guidance", true),
            ("-de", "--device", true),
            ("-a", "--attention-slicing", true),
        ],
        Command::Import => &[
            ("-m", "--model", true),
            ("-o", "--ollama", true),
            ("-hr", "--hf-dir", true),
            ("-or", "--ollama-dir", true),
            ("-ot", "--out", true),
        ],
        Command::Create => &[("-f", "--file", true), ("-o", "--ollama-dir", true)],
        Command::Delete => &[
            ("-m", "--model", true),
            ("-s", "--source", true),
            ("-o", "--ollama-dir", true),
            ("-y", "--yes", false),
        ],
        Command::Quantize => &[
            ("-t", "--type", true),
            ("-ts", "--threads", true),
            ("-o", "--out", true),
            ("-p", "--pure", false),
            ("-l", "--leave-output-tensor", false),
            ("-a", "--allow-requantize", false),
            ("-f", "--force", false),
        ],
        Command::Update => &[("-c", "--check", false), ("-v", "--version", true)],
        Command::Transcribe => &[
            ("-m", "--model", true),
            ("-a", "--audio", true),
            ("-l", "--language", true),
            ("-t", "--threads", true),
            ("-g", "--gpu", true),
            ("-j", "--json", false),
        ],
    }
}

pub(super) fn number(value: &str, name: &str, allow_zero: bool) -> Result<u64, String> {
    let result = value.parse::<u64>().ok().filter(|n| allow_zero || *n > 0);
    result.ok_or_else(|| {
        format!(
            "{name} needs {} integer, got '{value}'",
            if allow_zero {
                "a non-negative"
            } else {
                "a positive"
            }
        )
    })
}

pub(super) fn size(value: &str, name: &str, allow_zero: bool) -> Result<String, String> {
    let lower = value.to_ascii_lowercase();
    let (digits, multiplier) = if let Some(digits) = lower.strip_suffix('k') {
        (digits, 1024)
    } else if let Some(digits) = lower.strip_suffix('m') {
        (digits, 1024 * 1024)
    } else {
        (lower.as_str(), 1)
    };
    number(digits, name, allow_zero)?
        .checked_mul(multiplier)
        .filter(|n| *n <= i32::MAX as u64)
        .map(|n| n.to_string())
        .ok_or_else(|| format!("{name} is too large (maximum {})", i32::MAX))
}

pub(super) fn threads(value: &str) -> Result<String, String> {
    if value == "auto" {
        Ok(std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .to_string())
    } else {
        size(value, "--threads", false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_short_flag_table_rewrites_values_and_switches_for_every_command() {
        use Command::*;
        let families: &[(&[Command], &str, &str)] = &[
            (
                &[Serve, Chat, Launch],
                "m:model p:path o:ollama or:ollama-dir c:ctx-size t:threads ts:tokens g:gpu b:batch-size mh:ubatch-size f:flash-attn pr:projector ce:chat-template k:kv-cache mk:memory-fallback e:expert-cache-mib a:api ct:client ht:host pt:port ay:api-key tt:target",
                "pe:prompt-cache n:no-prompt-cache i:interactive",
            ),
            (&[Stop], "m:model ht:host p:port a:api-key", ""),
            (&[Update], "v:version", "c:check"),
            (&[List], "dr:dir o:ollama-dir", "j:json"),
            (
                &[Fetch],
                "f:file o:out i:index",
                "l:list oe:ollama-database le:lm-studio-database",
            ),
            (
                &[Image],
                "m:model p:prompt n:negative-prompt o:out t:tokenizer w:width ht:height s:steps sd:seed g:guidance de:device a:attention-slicing",
                "",
            ),
            (
                &[Import],
                "m:model o:ollama hr:hf-dir or:ollama-dir ot:out",
                "",
            ),
            (&[Create], "f:file o:ollama-dir", ""),
            (&[Delete], "m:model s:source o:ollama-dir", "y:yes"),
            (
                &[Quantize],
                "t:type ts:threads o:out",
                "p:pure l:leave-output-tensor a:allow-requantize f:force",
            ),
            (
                &[Transcribe],
                "m:model a:audio l:language t:threads g:gpu",
                "j:json",
            ),
        ];
        for &(commands, values, switches) in families {
            for &command in commands {
                let mut expected_count = 0;
                for mapping in values.split_whitespace() {
                    expected_count += 1;
                    let (short, long) = mapping.split_once(':').unwrap();
                    for value in ["plain", "name--with--dashes", "x=y"] {
                        for argv in [
                            vec![format!("-{short}").into(), OsString::from(value)],
                            vec![format!("-{short}={value}").into()],
                        ] {
                            let actual = short_arguments(command, Args::new(argv)).unwrap().rest();
                            assert_eq!(actual, vec![OsString::from(format!("--{long}={value}"))]);
                        }
                    }
                    let actual =
                        short_arguments(command, Args::new([format!("-{short}=--help").into()]))
                            .unwrap()
                            .rest();
                    assert_eq!(actual, vec![OsString::from(format!("--{long}=--help"))]);
                    for argv in [
                        vec![format!("-{short}")],
                        vec![format!("-{short}=")],
                        vec![format!("-{short}"), "--help".into()],
                    ] {
                        assert!(
                            short_arguments(
                                command,
                                Args::new(argv.into_iter().map(OsString::from))
                            )
                            .is_err()
                        );
                    }
                    assert!(
                        short_arguments(command, Args::new([format!("--{long}=value").into()]))
                            .is_err()
                    );
                }
                for mapping in switches.split_whitespace().chain(["h:help", "d:dry-run"]) {
                    let (short, long) = mapping.split_once(':').unwrap();
                    let actual = short_arguments(command, Args::new([format!("-{short}").into()]))
                        .unwrap()
                        .rest();
                    assert_eq!(actual, vec![OsString::from(format!("--{long}"))]);
                    assert!(
                        short_arguments(command, Args::new([format!("-{short}=true").into()]))
                            .is_err()
                    );
                    assert!(
                        short_arguments(command, Args::new([format!("--{long}").into()])).is_err()
                    );
                }
                // The independent contract above must cover the complete implementation table.
                assert_eq!(
                    short_flags(command).len(),
                    expected_count + switches.split_whitespace().count()
                );
                for alias in [
                    "--", "-q", "-od", "-V", "-context", "-threads", "-help", "-dry-run", "-z",
                ] {
                    assert!(
                        short_arguments(command, Args::new([alias.into()])).is_err(),
                        "accepted {alias}"
                    );
                }
            }
        }
    }

    #[test]
    fn normalization_consumes_values_before_looking_for_flags_and_validates_past_help() {
        let normalized = short_arguments(
            Command::Image,
            Args::new(
                [
                    "sd15",
                    "-p",
                    "a prompt containing --help",
                    "-n=--negative",
                    "-o",
                    "out--name.png",
                    "-d",
                ]
                .map(OsString::from),
            ),
        )
        .unwrap()
        .rest();
        assert_eq!(
            normalized,
            [
                "sd15",
                "--prompt=a prompt containing --help",
                "--negative-prompt=--negative",
                "--out=out--name.png",
                "--dry-run",
            ]
            .map(OsString::from)
        );
        for argv in [
            vec!["-h", "--help"],
            vec!["-h", "--"],
            vec!["-h", "-m"],
            vec!["-m=--", "-h"],
            vec!["-h", "-m", "--help"],
        ] {
            assert!(
                short_arguments(
                    Command::Serve,
                    Args::new(argv.into_iter().map(OsString::from))
                )
                .is_err()
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn flag_values_preserve_non_utf8_and_dash_prefixed_values_require_inline_form() {
        use std::os::unix::ffi::OsStringExt;
        let value = OsString::from_vec(b"folder--help-\xff".to_vec());
        for inline in [false, true] {
            let mut argument = OsString::from("-m");
            let mut argv = Vec::new();
            if inline {
                argument.push("=");
                argument.push(&value);
                argv.push(argument);
            } else {
                argv.extend([argument, value.clone()]);
            }
            let mut normalized = short_arguments(Command::Serve, Args::new(argv)).unwrap();
            let argument = normalized.next().unwrap();
            assert_eq!(argument, "--model");
            let flag = Flag::parse(&argument).unwrap();
            assert_eq!(flag.name, "--model");
            assert_eq!(normalized.value(&flag).unwrap(), value);
            assert!(normalized.is_empty());
        }
        let value = OsString::from_vec(b"--\xff".to_vec());
        assert!(short_arguments(Command::Serve, Args::new(["-m".into(), value.clone()])).is_err());
        let mut inline = OsString::from("-m=");
        inline.push(&value);
        let mut normalized = short_arguments(Command::Serve, Args::new([inline])).unwrap();
        let flag = Flag::parse(&normalized.next().unwrap()).unwrap();
        assert_eq!(normalized.value(&flag).unwrap(), value);
    }

    #[test]
    fn sizes_are_checked_and_use_binary_units() {
        assert_eq!(size("8k", "context", true).unwrap(), "8192");
        assert_eq!(size("1M", "context", true).unwrap(), "1048576");
        assert_eq!(size("0", "context", true).unwrap(), "0");
        for value in ["-1", "8.5k", "xyz", "18446744073709551615m", "2147483648"] {
            assert!(size(value, "context", true).is_err());
        }
        assert!(size("0", "batch", false).is_err());
    }

    #[test]
    fn automatic_threads_is_positive() {
        assert!(threads("auto").unwrap().parse::<usize>().unwrap() > 0);
    }
}
