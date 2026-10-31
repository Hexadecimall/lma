use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
};

fn lma() -> Command {
    Command::new(env!("CARGO_BIN_EXE_lma"))
}

fn isolated_lma(temp: &TempDir) -> Command {
    let mut command = lma();
    command.current_dir(&temp.0).env("PATH", "");
    command
}

fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).unwrap()
}
fn stderr(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).unwrap()
}

struct TempDir(PathBuf);
impl TempDir {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "lma-cli-test-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn color_never_leaks_into_pipes_or_machine_output() {
    let temp = TempDir::new();
    for (term, no_color) in [("xterm-256color", ""), ("xterm", "0"), ("dumb", "")] {
        for argv in [
            vec!["h"],
            vec!["-v"],
            vec!["l"],
            vec!["l", "-j"],
            vec!["l", "-d"],
            vec!["c", "example", "-d"],
            vec!["unknown-command"],
            vec!["q", "missing.gguf"],
        ] {
            let output = isolated_lma(&temp)
                .env_clear()
                .env("TERM", term)
                .env("NO_COLOR", no_color)
                .env("FORCE_COLOR", "1")
                .env("CLICOLOR_FORCE", "1")
                .args(&argv)
                .output()
                .unwrap();
            assert!(
                !output.stdout.contains(&0x1b),
                "{argv:?}: {}",
                stdout(&output)
            );
            assert!(
                !output.stderr.contains(&0x1b),
                "{argv:?}: {}",
                stderr(&output)
            );
            if argv.contains(&"-j") || argv.contains(&"-d") {
                assert!(output.status.success(), "{}", stderr(&output));
                serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap();
            }
        }
    }
    let output = isolated_lma(&temp)
        .arg("bad\x1b]8;;https://example.org\x07\r\n\u{202e}")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let error = stderr(&output);
    assert!(!error.contains('\x1b') && !error.contains('\x07') && !error.contains('\u{202e}'));
    assert!(error.contains("\\u{1b}") && error.contains("\\u{202e}"));
}

// The optional PTY driver uses Python's standard library, not a shell or new crate.
// Pure Rust tests above and in terminal_style still run on every platform.
#[cfg(unix)]
#[derive(Clone, Copy)]
struct TerminalOptions {
    stdout: bool,
    stderr: bool,
    width: u16,
    term: &'static str,
    no_color: Option<&'static str>,
}

#[cfg(unix)]
impl Default for TerminalOptions {
    fn default() -> Self {
        Self {
            stdout: true,
            stderr: true,
            width: 80,
            term: "xterm-256color",
            no_color: None,
        }
    }
}

#[cfg(unix)]
#[derive(serde::Deserialize)]
struct TerminalOutput {
    code: i32,
    stdout: String,
    stderr: String,
}

#[cfg(unix)]
fn terminal_lma(temp: &TempDir, argv: &[&str], options: TerminalOptions) -> Option<TerminalOutput> {
    const DRIVER: &str = r#"
import errno, fcntl, json, os, pty, struct, subprocess, sys, tempfile, termios, threading
binary, cwd, settings = sys.argv[1:4]
settings = json.loads(settings)
env = {
    'PATH': '', 'TERM': settings['term'], 'HOME': cwd + '/missing-home',
    'USERPROFILE': cwd + '/missing-home', 'OLLAMA_MODELS': cwd + '/missing-ollama',
}
if settings['no_color'] is not None:
    env['NO_COLOR'] = settings['no_color']
streams, masters, results = [], [], [bytearray(), bytearray()]
for enabled in (settings['stdout'], settings['stderr']):
    if enabled:
        master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 24, settings['width'], 0, 0))
        streams.append(os.fdopen(slave, 'wb', buffering=0))
        masters.append(master)
    else:
        streams.append(tempfile.TemporaryFile())
        masters.append(None)
def claim_terminal():
    for stream, master in zip(streams, masters):
        if master is not None:
            fcntl.ioctl(stream.fileno(), termios.TIOCSCTTY, 0)
            break
def read_terminal(index, master):
    while True:
        try:
            chunk = os.read(master, 65536)
        except OSError as error:
            if error.errno == errno.EIO:
                break
            raise
        if not chunk:
            break
        results[index].extend(chunk)
child = subprocess.Popen([binary] + sys.argv[4:], cwd=cwd, env=env,
    stdin=subprocess.DEVNULL, stdout=streams[0], stderr=streams[1],
    start_new_session=True, preexec_fn=claim_terminal)
threads = []
for index, master in enumerate(masters):
    if master is not None:
        streams[index].close()
        thread = threading.Thread(target=read_terminal, args=(index, master), daemon=True)
        thread.start()
        threads.append(thread)
try:
    code = child.wait(timeout=15)
finally:
    if child.poll() is None:
        child.kill()
        child.wait()
for thread in threads:
    thread.join(timeout=5)
    assert not thread.is_alive(), 'PTY output reader timed out'
for index, master in enumerate(masters):
    if master is None:
        streams[index].seek(0)
        results[index].extend(streams[index].read())
        streams[index].close()
    else:
        os.close(master)
print(json.dumps(dict(code=code,
    stdout=results[0].decode('utf-8').replace('\r\n', '\n'),
    stderr=results[1].decode('utf-8').replace('\r\n', '\n'))))
"#;
    let settings = serde_json::json!({
        "stdout": options.stdout, "stderr": options.stderr, "width": options.width,
        "term": options.term, "no_color": options.no_color,
    });
    let output = match Command::new("python3")
        .args(["-c", DRIVER, env!("CARGO_BIN_EXE_lma")])
        .arg(&temp.0)
        .arg(settings.to_string())
        .args(argv)
        .output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("skipping optional PTY color test: python3 is unavailable");
            return None;
        }
        Err(error) => panic!("cannot run PTY test driver: {error}"),
    };
    assert!(output.status.success(), "PTY driver: {}", stderr(&output));
    Some(serde_json::from_slice(&output.stdout).unwrap())
}

#[cfg(unix)]
fn without_sgr(text: &str) -> String {
    let mut plain = String::new();
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c != '\x1b' {
            plain.push(c);
            continue;
        }
        assert_eq!(chars.next(), Some('['), "only styling SGR is allowed");
        loop {
            match chars.next() {
                Some('m') => break,
                Some(c) if c.is_ascii_digit() || c == ';' => {}
                other => panic!("unexpected terminal escape: {other:?}"),
            }
        }
    }
    plain
}

#[cfg(unix)]
#[test]
fn terminal_color_respects_each_stream_no_color_and_dumb() {
    let temp = TempDir::new();
    for options in [
        TerminalOptions::default(),
        TerminalOptions {
            no_color: Some(""),
            ..Default::default()
        },
        TerminalOptions {
            no_color: Some("0"),
            ..Default::default()
        },
        TerminalOptions {
            no_color: Some("1"),
            ..Default::default()
        },
        TerminalOptions {
            term: "dumb",
            ..Default::default()
        },
        TerminalOptions {
            stdout: false,
            ..Default::default()
        },
        TerminalOptions {
            stderr: false,
            ..Default::default()
        },
    ] {
        let color = options.term != "dumb" && !options.no_color.is_some_and(|v| !v.is_empty());
        for argv in [vec!["h"], vec!["-v"], vec!["nss", "chat", "--help"]] {
            let Some(output) = terminal_lma(&temp, &argv, options) else {
                return;
            };
            assert_eq!(output.code, 0, "{}", output.stderr);
            assert_eq!(output.stdout.contains('\x1b'), color && options.stdout);
            let piped = isolated_lma(&temp)
                .env_clear()
                .args(&argv)
                .output()
                .unwrap();
            assert_eq!(without_sgr(&output.stdout), stdout(&piped));
            assert!(output.stderr.is_empty());
        }
        for (argv, code) in [(vec!["unknown-command"], 2), (vec!["q", "missing.gguf"], 1)] {
            let Some(output) = terminal_lma(&temp, &argv, options) else {
                return;
            };
            assert_eq!(output.code, code, "{}", output.stderr);
            assert_eq!(output.stderr.contains('\x1b'), color && options.stderr);
            assert!(without_sgr(&output.stderr).starts_with("lma: "));
            assert!(output.stdout.is_empty());
        }
    }
}

#[cfg(unix)]
#[test]
fn terminal_model_lists_fit_width_and_keep_json_and_warnings_separate() {
    use unicode_width::UnicodeWidthStr;
    let temp = TempDir::new();
    let models = temp.0.join("models");
    fs::create_dir(&models).unwrap();
    let filename = "模型-very-long-model-name-that-needs-wrapping.gguf";
    fs::write(models.join(filename), b"GGUFfixture").unwrap();
    let warning_file = "not-a-directory\x1b[2J\u{202e}";
    fs::write(temp.0.join(warning_file), b"fixture").unwrap();
    for width in [24, 60, 80, 120] {
        let options = TerminalOptions {
            width,
            ..Default::default()
        };
        let Some(output) = terminal_lma(&temp, &["l"], options) else {
            return;
        };
        assert_eq!(output.code, 0, "{}", output.stderr);
        assert!(output.stdout.contains('\x1b'));
        let plain = without_sgr(&output.stdout);
        assert!(
            plain.lines().all(|line| line.width() <= usize::from(width)),
            "{width}: {plain}"
        );
        if width < 76 {
            assert!(plain.starts_with("INSTALLED MODELS"));
            assert!(!plain.contains('…'));
        } else {
            assert!(plain.starts_with("NAME"));
            assert!(plain.contains("SOURCE") && plain.contains("SIZE") && plain.contains("MODEL"));
            assert!(plain.contains('…'));
        }
    }
    for options in [
        TerminalOptions::default(),
        TerminalOptions {
            no_color: Some("1"),
            ..Default::default()
        },
        TerminalOptions {
            stdout: false,
            ..Default::default()
        },
        TerminalOptions {
            stderr: false,
            ..Default::default()
        },
        TerminalOptions {
            term: "dumb",
            ..Default::default()
        },
    ] {
        let Some(output) = terminal_lma(&temp, &["l", "-dr", warning_file], options) else {
            return;
        };
        assert_eq!(output.code, 0, "{}", output.stderr);
        let color = options.term != "dumb" && options.no_color.is_none();
        assert_eq!(output.stdout.contains('\x1b'), color && options.stdout);
        assert_eq!(output.stderr.contains('\x1b'), color && options.stderr);
        let warning = without_sgr(&output.stderr);
        assert!(warning.contains("not a directory"));
        assert!(warning.contains("\\u{1b}") && warning.contains("\\u{202e}"));
        assert!(!warning.contains('\u{202e}'));
        if !options.stdout {
            assert!(output.stdout.starts_with("NAME"));
            assert!(output.stdout.contains(filename));
        }
        for argv in [vec!["l", "-j", "-dr", warning_file], vec!["l", "-d"]] {
            let Some(output) = terminal_lma(&temp, &argv, options) else {
                return;
            };
            assert_eq!(output.code, 0, "{}", output.stderr);
            assert!(!output.stdout.contains('\x1b'));
            assert!(output.stderr.is_empty());
            let json: serde_json::Value = serde_json::from_str(&output.stdout).unwrap();
            if argv.contains(&"-j") {
                assert_eq!(json["models"][0]["name"], filename);
                assert!(!json["warnings"].as_array().unwrap().is_empty());
            }
        }
    }
}

#[cfg(unix)]
#[test]
fn terminal_model_names_escape_controls_but_json_preserves_original_data() {
    let temp = TempDir::new();
    let models = temp.0.join("models");
    fs::create_dir(&models).unwrap();
    let filename = "unsafe\x1b[2J\u{202e}\n.gguf";
    fs::write(models.join(filename), b"GGUFfixture").unwrap();
    for options in [
        TerminalOptions {
            width: 240,
            ..Default::default()
        },
        TerminalOptions {
            stdout: false,
            ..Default::default()
        },
    ] {
        let Some(output) = terminal_lma(&temp, &["l"], options) else {
            return;
        };
        assert_eq!(output.code, 0, "{}", output.stderr);
        let plain = without_sgr(&output.stdout);
        assert!(
            plain.contains("\\u{1b}") && plain.contains("\\u{202e}"),
            "{plain}"
        );
        assert!(!plain.contains('\u{202e}'));
        assert!(!plain.chars().any(|c| c.is_control() && c != '\n'));
        let Some(output) = terminal_lma(&temp, &["l", "-j"], options) else {
            return;
        };
        assert_eq!(output.code, 0, "{}", output.stderr);
        assert!(!output.stdout.contains('\x1b'));
        let json: serde_json::Value = serde_json::from_str(&output.stdout).unwrap();
        assert_eq!(json["models"][0]["name"], filename);
    }
}

#[test]
fn help_and_version_need_no_external_executable() {
    for argv in [
        vec![],
        vec!["nss", "help"],
        vec!["nss", "help", "serve"],
        vec!["nss", "help", "serve", "advanced"],
        vec!["nss", "help", "q", "--all"],
        vec!["nss", "serve", "--help-all"],
        vec!["nss", "quant", "--help"],
        vec!["nss", "help", "launch"],
        vec!["nss", "help", "lh"],
        vec!["nss", "launch", "--help"],
        vec!["nss", "lh", "-h"],
        vec!["nss", "help", "l", "advanced"],
        vec!["nss", "help", "chat", "--all"],
        vec!["nss", "launch", "--help-all"],
        vec!["nss", "list", "-h"],
        vec!["nss", "chat", "--help"],
        vec!["nss", "imagen", "--help"],
        vec!["nss", "import", "--help"],
        vec!["nss", "create", "--help"],
        vec!["nss", "help", "create"],
        vec!["nss", "create", "-h"],
        vec!["nss", "help", "create"],
        vec!["nss", "help", "import"],
        vec!["nss", "import", "-h"],
        vec!["nss", "transcribe", "--help"],
        vec!["nss", "help", "transcribe"],
        vec!["nss", "help", "transcribe", "advanced"],
        vec!["nss", "transcribe", "-h"],
        vec!["nss", "launch", "-h"],
        vec!["nss", "chat", "--help"],
        vec!["nss", "--help"],
        vec!["nss", "--version"],
    ] {
        let output = lma().env("PATH", "").args(argv).output().unwrap();
        assert!(output.status.success(), "{}", stderr(&output));
        assert!(stdout(&output).contains("lma"));
    }
}

#[test]
fn short_commands_pair_with_nss_and_removed_long_forms_are_actionable() {
    for (short, long) in [
        ("h", "help"),
        ("l", "list"),
        ("lh", "launch"),
        ("c", "chat"),
        ("t", "transcribe"),
        ("s", "serve"),
        ("sp", "stop"),
        ("q", "quantize"),
        ("f", "fetch"),
        ("i", "image"),
        ("it", "import"),
        ("ce", "create"),
        ("d", "delete"),
    ] {
        let short_output = lma().env("PATH", "").args([short, "-h"]).output().unwrap();
        let long_output = lma()
            .env("PATH", "")
            .args(["nss", long, "--help"])
            .output()
            .unwrap();
        assert!(
            short_output.status.success(),
            "{short}: {}",
            stderr(&short_output)
        );
        assert!(
            long_output.status.success(),
            "{long}: {}",
            stderr(&long_output)
        );
        assert_eq!(stdout(&short_output), stdout(&long_output));
    }
    let temp = TempDir::new();
    for old in [
        "help",
        "serve",
        "server",
        "stop",
        "launch",
        "list",
        "chat",
        "transcribe",
        "quantize",
        "quant",
        "fetch",
        "image",
        "imagen",
        "image-gen",
        "img",
        "import",
        "create",
        "delete",
    ] {
        let output = isolated_lma(&temp).args([old, "--help"]).output().unwrap();
        assert_eq!(output.status.code(), Some(2), "{old}: {}", stderr(&output));
        assert!(stderr(&output).contains(&format!("lma nss {old}")));
        assert!(stderr(&output).contains("lma nss help"));
        assert!(stdout(&output).is_empty());
        let help = isolated_lma(&temp)
            .args(["nss", old, "--help"])
            .output()
            .unwrap();
        assert!(help.status.success(), "{old}: {}", stderr(&help));
        assert!(stdout(&help).contains("lma"));
        assert!(stderr(&help).is_empty());
        assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
    }
    let help = isolated_lma(&temp)
        .args(["nss", "lh", "--help"])
        .output()
        .unwrap();
    assert!(help.status.success(), "{}", stderr(&help));
    assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
}

#[test]
fn canonical_root_dry_runs_match_explicit_nss_configuration_for_every_family() {
    let temp = TempDir::new();
    let cases: &[(&[&str], &[&str], &str)] = &[
        (
            &[
                "s",
                "-p",
                "missing.gguf",
                "-c",
                "8k",
                "-t",
                "2",
                "-g",
                "cpu",
                "-b",
                "32",
                "-mh",
                "8",
                "-f",
                "on",
                "-pr",
                "projector.gguf",
                "-ce",
                "plain",
                "-pe",
                "-n",
                "-k",
                "q8_0",
                "-mk",
                "mmap",
                "-a",
                "openai",
                "-ct",
                "zed",
                "-ht",
                "localhost",
                "-pt",
                "9001",
                "-ay",
                "parity-secret",
            ],
            &[
                "serve",
                "--path",
                "missing.gguf",
                "--ctx-size",
                "8k",
                "--threads",
                "2",
                "--gpu",
                "cpu",
                "--batch-size",
                "32",
                "--ubatch-size",
                "8",
                "--flash-attn",
                "on",
                "--projector",
                "projector.gguf",
                "--chat-template",
                "plain",
                "--prompt-cache",
                "--no-prompt-cache",
                "--kv-cache",
                "q8_0",
                "--memory-fallback",
                "mmap",
                "--api",
                "openai",
                "--client",
                "zed",
                "--host",
                "localhost",
                "--port",
                "9001",
                "--api-key",
                "parity-secret",
            ],
            "serve",
        ),
        (
            &[
                "c",
                "-m=model.gguf",
                "-or=missing-store",
                "-g=cpu",
                "-e=16",
                "-i",
            ],
            &[
                "chat",
                "--model=model.gguf",
                "--ollama-dir=missing-store",
                "--gpu=cpu",
                "--expert-cache-mib=16",
                "--interactive",
            ],
            "launch",
        ),
        (
            &[
                "lh",
                "-o=local:latest",
                "-or=missing-store",
                "-tt=codex",
                "-ct=zed",
                "-a=ollama",
                "-i",
            ],
            &[
                "launch",
                "--ollama=local:latest",
                "--ollama-dir=missing-store",
                "--target=codex",
                "--client=zed",
                "--api=ollama",
                "--interactive",
            ],
            "launch",
        ),
        (
            &[
                "sp",
                "-m=local:latest",
                "-ht=localhost",
                "-p=9002",
                "-a=parity-secret",
            ],
            &[
                "stop",
                "--model=local:latest",
                "--host=localhost",
                "--port=9002",
                "--api-key=parity-secret",
            ],
            "stop",
        ),
        (
            &[
                "l",
                "tiny",
                "-j",
                "-dr=first",
                "-dr",
                "second",
                "-o=missing-store",
            ],
            &[
                "list",
                "tiny",
                "--json",
                "--dir=first",
                "--dir",
                "second",
                "--ollama-dir=missing-store",
            ],
            "list",
        ),
        (
            &[
                "q",
                "in.gguf",
                "-t=q8_0",
                "-ts=2",
                "-o=out.gguf",
                "-p",
                "-l",
                "-a",
                "-f",
            ],
            &[
                "quantize",
                "in.gguf",
                "--type=q8_0",
                "--threads=2",
                "--out=out.gguf",
                "--pure",
                "--leave-output-tensor",
                "--allow-requantize",
                "--force",
            ],
            "quantize",
        ),
        (
            &[
                "f",
                "owner/repo",
                "-oe",
                "-le",
                "-l",
                "-f=weights.gguf",
                "-o=downloads",
            ],
            &[
                "fetch",
                "owner/repo",
                "--ollama-database",
                "--lm-studio-database",
                "--list",
                "--file=weights.gguf",
                "--out=downloads",
            ],
            "fetch",
        ),
        (
            &[
                "i",
                "-m=sd15",
                "-p=a fox",
                "-n=blurry",
                "-o=fox.png",
                "-t=tokenizer.json",
                "-w=64",
                "-ht=128",
                "-s=3",
                "-sd=0",
                "-g=1",
                "-de=cpu",
                "-a=8",
            ],
            &[
                "image",
                "--model=sd15",
                "--prompt=a fox",
                "--negative-prompt=blurry",
                "--out=fox.png",
                "--tokenizer=tokenizer.json",
                "--width=64",
                "--height=128",
                "--steps=3",
                "--seed=0",
                "--guidance=1",
                "--device=cpu",
                "--attention-slicing=8",
            ],
            "image",
        ),
        (
            &[
                "t",
                "-m=whisper",
                "-a=missing.wav",
                "-l=en",
                "-t=2",
                "-g=off",
                "-j",
            ],
            &[
                "transcribe",
                "--model=whisper",
                "--audio=missing.wav",
                "--language=en",
                "--threads=2",
                "--gpu=off",
                "--json",
            ],
            "transcribe",
        ),
    ];
    for &(root, nss, family) in cases {
        let run = |prefix: &[&str], argv: &[&str], dry_run: &str| {
            isolated_lma(&temp)
                .env_clear()
                .env("PATH", "")
                .env("HOME", temp.0.join("missing-home"))
                .env("OLLAMA_MODELS", temp.0.join("missing-store"))
                .stdin(std::process::Stdio::null())
                .args(prefix)
                .args(argv)
                .arg(dry_run)
                .output()
                .unwrap()
        };
        let short = run(&[], root, "-d");
        let long = run(&["nss"], nss, "--dry-run");
        assert!(short.status.success(), "{root:?}: {}", stderr(&short));
        assert!(long.status.success(), "{nss:?}: {}", stderr(&long));
        let short_plan: serde_json::Value = serde_json::from_str(&stdout(&short)).unwrap();
        let long_plan: serde_json::Value = serde_json::from_str(&stdout(&long)).unwrap();
        assert_eq!(short_plan, long_plan, "{root:?} vs {nss:?}");
        assert_eq!(short_plan["command"], family);
        assert_eq!(short_plan["dry_run"], true);
        if root[0] == "c" || root[0] == "lh" {
            assert_eq!(
                short_plan["options"]["mode"],
                if root[0] == "c" { "chat" } else { "launch" }
            );
            assert_eq!(short_plan["options"]["interactive"], true);
        }
        assert!(!stdout(&short).contains("parity-secret"));
        assert!(!stdout(&long).contains("parity-secret"));
        assert_eq!(stderr(&short), stderr(&long));
        assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
    }
}

#[test]
fn canonical_create_and_import_reach_the_same_runtime_validation_as_nss() {
    let temp = TempDir::new();
    fs::create_dir(temp.0.join("local-hf")).unwrap();
    let cases: &[(&[&str], &[&str], &str)] = &[
        (
            &[
                "ce",
                "local",
                "-f=missing-Modelfile",
                "-o=missing-store",
                "-d",
            ],
            &[
                "nss",
                "create",
                "local",
                "--file=missing-Modelfile",
                "--ollama-dir=missing-store",
                "--dry-run",
            ],
            "missing-Modelfile",
        ),
        (
            &[
                "it",
                "-o=local:latest",
                "-or=missing-store",
                "-ot=new.gguf",
                "-d",
            ],
            &[
                "nss",
                "import",
                "--ollama=local:latest",
                "--ollama-dir=missing-store",
                "--out=new.gguf",
                "--dry-run",
            ],
            "cannot access Ollama model store",
        ),
        (
            &[
                "it",
                "-m=local:latest",
                "-or=missing-store",
                "-ot=new.gguf",
                "-d",
            ],
            &[
                "nss",
                "import",
                "--model=local:latest",
                "--ollama-dir=missing-store",
                "--out=new.gguf",
                "--dry-run",
            ],
            "cannot access Ollama model store",
        ),
        (
            &["it", "-hr=local-hf", "-ot=new.gguf", "-d"],
            &[
                "nss",
                "import",
                "--hf-dir=local-hf",
                "--out=new.gguf",
                "--dry-run",
            ],
            "config.json",
        ),
    ];
    for &(root, nss, expected) in cases {
        let run = |argv: &[&str]| {
            isolated_lma(&temp)
                .env_clear()
                .env("PATH", "")
                .env("HOME", temp.0.join("missing-home"))
                .args(argv)
                .output()
                .unwrap()
        };
        let short = run(root);
        let long = run(nss);
        assert_eq!(short.status.code(), Some(1), "{root:?}: {}", stderr(&short));
        assert_eq!(long.status.code(), Some(1), "{nss:?}: {}", stderr(&long));
        assert_eq!(stderr(&short), stderr(&long));
        assert!(
            stderr(&short).contains(expected),
            "{root:?}: {}",
            stderr(&short)
        );
        assert!(stdout(&short).is_empty());
        assert!(stdout(&long).is_empty());
        assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 1);
        assert_eq!(fs::read_dir(temp.0.join("local-hf")).unwrap().count(), 0);
    }
}

#[test]
fn root_rejects_all_long_flags_and_legacy_option_aliases_even_after_help() {
    let temp = TempDir::new();
    let host_flags = "--model --path --ollama --ollama-dir --ctx-size --context --threads --gpu --n-gpu-layers --batch --batch-size --micro-batch --ubatch-size --flash --flash-attn --projector --mmproj --chat-template --prompt-cache --no-prompt-cache --kv-cache --memory-fallback --expert-cache-mib --api --client --host --port --api-key --target --interactive --mode -od";
    for (family, flags) in [
        ("s", host_flags),
        ("c", host_flags),
        ("lh", host_flags),
        ("sp", "--model --host --port --api-key"),
        ("l", "--json --dir --ollama-dir -od"),
        (
            "q",
            "--type --threads --out --pure --leave-output --leave-output-tensor --allow-requantize --force -q",
        ),
        (
            "f",
            "--list --file --out --output --ollama-database --lm-studio-database",
        ),
        (
            "i",
            "--model --model-dir --prompt --negative-prompt --out --output --tokenizer --width --height --steps --seed --guidance --device --attention-slicing",
        ),
        (
            "it",
            "--model --ollama --ollama-dir --hf-dir --out --output -od",
        ),
        ("ce", "--file --ollama-dir -od"),
        ("t", "--model --audio --language --threads --gpu --json"),
    ] {
        for flag in flags.split_whitespace().chain([
            "--",
            "--help",
            "--help-all",
            "--version",
            "--dry-run",
            "--future-option",
            "-V",
        ]) {
            for argv in [
                vec![family, flag],
                vec![family, "-h", flag],
                vec![family, flag, "-h"],
            ] {
                let output = isolated_lma(&temp).args(&argv).output().unwrap();
                assert_eq!(
                    output.status.code(),
                    Some(2),
                    "{argv:?}: {}",
                    stdout(&output)
                );
                let error = stderr(&output);
                assert!(
                    error.contains("unsupported root option"),
                    "{argv:?}: {error}"
                );
                assert!(error.contains(flag), "{argv:?}: {error}");
                assert!(error.contains("lma nss help"), "{argv:?}: {error}");
                assert!(stdout(&output).is_empty());
            }
        }
    }
    for flag in ["--help", "--help-all", "--version", "--dry-run", "--", "-V"] {
        let output = isolated_lma(&temp).arg(flag).output().unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(stdout(&output).is_empty());
        assert!(stderr(&output).contains("lma"));
    }
    assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
}

#[test]
fn stop_previews_are_local_redacted_and_do_not_start_or_contact_a_host() {
    let temp = TempDir::new();
    for prefix in [
        vec!["nss", "stop"],
        vec!["nss", "stop"],
        vec!["nss", "stop"],
        vec!["nss", "--dry-run", "stop"],
    ] {
        let mut command = isolated_lma(&temp);
        command.args(&prefix).args([
            "-m=llama3.2",
            "--host=localhost",
            "--port=9000",
            "--api-key=private-stop-key",
        ]);
        if !prefix.contains(&"--dry-run") {
            command.arg("--dry-run");
        }
        let output = command.output().unwrap();
        assert!(output.status.success(), "{}", stderr(&output));
        let plan: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
        assert_eq!(plan["command"], "stop");
        assert_eq!(plan["options"]["model"], "llama3.2");
        assert_eq!(plan["options"]["host"], "localhost");
        assert_eq!(plan["options"]["port"], 9000);
        assert_eq!(plan["options"]["api_key"], "[redacted]");
        assert!(!stdout(&output).contains("private-stop-key"));
    }
    let output = isolated_lma(&temp)
        .args(["nss", "stop", "--dry-run"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let plan: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert!(plan["options"]["model"].is_null());
    assert_eq!(plan["options"]["host"], "127.0.0.1");
    assert_eq!(plan["options"]["port"], 8080);
    let output = isolated_lma(&temp)
        .args(["nss", "stop", "model.gguf", "--dry-run"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("lma s -p FILE"));
    assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
}

#[test]
fn explicit_paths_are_distinct_from_installed_names_and_modes_are_removed() {
    let temp = TempDir::new();
    for prefix in [
        vec!["nss", "serve"],
        vec!["nss", "chat"],
        vec!["nss", "launch"],
        vec!["nss", "serve"],
        vec!["nss", "chat"],
    ] {
        for (flag, expected) in [
            ("--model", PathBuf::from("tiny")),
            ("--path", PathBuf::from(".").join("tiny")),
        ] {
            let output = isolated_lma(&temp)
                .args(&prefix)
                .args([flag, "tiny", "--dry-run"])
                .output()
                .unwrap();
            assert!(output.status.success(), "{}", stderr(&output));
            let plan: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
            assert_eq!(plan["options"]["model"], serde_json::json!(expected));
        }
        for mode in ["chat", "host"] {
            let output = isolated_lma(&temp)
                .args(&prefix)
                .args(["--mode", mode, "--dry-run"])
                .output()
                .unwrap();
            assert_eq!(output.status.code(), Some(2));
            assert!(stderr(&output).contains("--mode has been removed"));
        }
    }
    let absolute = temp.0.join("not-loaded.gguf");
    let output = isolated_lma(&temp)
        .args(["nss", "serve", "--path"])
        .arg(&absolute)
        .arg("--dry-run")
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let plan: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_eq!(plan["options"]["model"], serde_json::json!(absolute));
    assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
}

#[test]
fn modelfile_create_preview_registration_and_profile_resolution_never_load_or_copy_weights() {
    let temp = TempDir::new();
    let home = temp.0.join("home");
    fs::create_dir(&home).unwrap();
    let weights = temp.0.join("weights.gguf");
    fs::write(&weights, b"GGUFfixture").unwrap();
    fs::write(temp.0.join("Modelfile"), "FROM ./weights.gguf\nSYSTEM Be concise.\nPARAMETER temperature 0\nPARAMETER num_ctx 256\nPARAMETER num_gpu 0\n").unwrap();
    let run = |args: &[&str]| {
        isolated_lma(&temp)
            .env("HOME", &home)
            .env("USERPROFILE", &home)
            .env_remove("LMA_MODELS")
            .env("OLLAMA_MODELS", home.join("missing-ollama"))
            .args(args)
            .output()
            .unwrap()
    };
    let preview = run(&["nss", "create", "custom", "--dry-run"]);
    assert!(preview.status.success(), "{}", stderr(&preview));
    let root_preview = run(&["ce", "custom", "-f", "Modelfile", "-d"]);
    assert!(root_preview.status.success(), "{}", stderr(&root_preview));
    let preview: serde_json::Value = serde_json::from_str(&stdout(&preview)).unwrap();
    let root_preview: serde_json::Value = serde_json::from_str(&stdout(&root_preview)).unwrap();
    assert_eq!(root_preview, preview);
    assert_eq!(preview["definition"]["system"], "Be concise.");
    assert!(!home.join(".lma").exists());
    assert!(
        run(&["nss", "create", "custom", "-f", "Modelfile"])
            .status
            .success()
    );
    let registry = home.join(".lma/models/.registry-v1.json");
    let before = fs::read(&registry).unwrap();
    assert!(!run(&["nss", "create", "custom"]).status.success());
    assert_eq!(fs::read(&registry).unwrap(), before);
    assert_eq!(fs::read_dir(registry.parent().unwrap()).unwrap().count(), 2);
    assert_eq!(fs::read(&weights).unwrap(), b"GGUFfixture");
    let resolved = run(&["nss", "serve", "lma:custom:latest", "--dry-run"]);
    assert!(resolved.status.success(), "{}", stderr(&resolved));
    let plan: serde_json::Value = serde_json::from_str(&stdout(&resolved)).unwrap();
    assert_eq!(plan["options"]["definition"]["system"], "Be concise.");
    assert_eq!(plan["options"]["context"], 256);
    assert_eq!(plan["options"]["gpu_layers"], 0);
    let override_context = run(&[
        "nss",
        "serve",
        "lma:custom:latest",
        "--context",
        "512",
        "--dry-run",
    ]);
    assert!(
        override_context.status.success(),
        "{}",
        stderr(&override_context)
    );
    let plan: serde_json::Value = serde_json::from_str(&stdout(&override_context)).unwrap();
    assert_eq!(plan["options"]["context"], 512);

    // A bare alias must not fall through to same-named Ollama weights, and an
    // explicit Ollama selector must not inherit an unrelated lma definition.
    let store = home.join("ollama-store");
    let manifest = store.join("manifests/registry.ollama.ai/library/custom/latest");
    let digest = "1".repeat(64);
    let blob = store.join("blobs").join(format!("sha256-{digest}"));
    fs::create_dir_all(manifest.parent().unwrap()).unwrap();
    fs::create_dir_all(blob.parent().unwrap()).unwrap();
    fs::write(&blob, b"GGUFother").unwrap();
    fs::write(&manifest, serde_json::to_vec(&serde_json::json!({
        "schemaVersion": 2,
        "layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": format!("sha256:{digest}"), "size": 9}]
    })).unwrap()).unwrap();
    for (selector, expected_weights, expected_system) in [
        ("custom", &weights, serde_json::json!("Be concise.")),
        ("ollama:custom", &blob, serde_json::Value::Null),
    ] {
        let output = run(&[
            "nss",
            "serve",
            selector,
            "--ollama-dir",
            store.to_str().unwrap(),
            "--dry-run",
        ]);
        assert!(output.status.success(), "{}", stderr(&output));
        let plan: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
        assert_eq!(
            plan["options"]["model"],
            serde_json::json!(fs::canonicalize(expected_weights).unwrap())
        );
        assert_eq!(plan["options"]["definition"]["system"], expected_system);
    }
    assert_eq!(fs::read(&weights).unwrap(), b"GGUFfixture");
    assert_eq!(fs::read(&blob).unwrap(), b"GGUFother");
}

#[test]
fn create_help_documents_the_supported_modelfile_subset_without_opening_files() {
    let temp = TempDir::new();
    for argv in [
        vec!["nss", "create", "--help"],
        vec!["nss", "help", "create", "advanced"],
        vec!["nss", "create", "-h"],
        vec!["nss", "help", "create"],
    ] {
        let output = isolated_lma(&temp).args(&argv).output().unwrap();
        assert!(output.status.success(), "{argv:?}: {}", stderr(&output));
        let help = stdout(&output);
        for expected in [
            "lma ce NAME",
            "lma nss create NAME",
            "-f PATH",
            "default: ./Modelfile",
            "FROM MODEL",
            "SYSTEM TEXT",
            "MESSAGE ROLE TEXT",
            "LICENSE TEXT",
            "TEMPLATE NAME",
            "temperature",
            "top_p",
            "seed",
            "num_predict",
            "num_ctx",
            "num_gpu",
            "num_thread",
            "num_batch",
            "u_batch",
            "Custom Ollama Go TEMPLATE",
            "rejected, not silently accepted",
            "1 MiB",
            "preview without registry writes",
            "not automatically discover",
        ] {
            assert!(
                help.contains(expected),
                "{argv:?}: missing {expected}: {help}"
            );
        }
        assert!(stderr(&output).is_empty());
        assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
    }
}

#[test]
fn create_argument_errors_are_rejected_before_modelfile_io() {
    let temp = TempDir::new();
    for argv in [
        vec!["nss", "create"],
        vec!["nss", "create", ""],
        vec!["nss", "create", "one", "two"],
        vec!["nss", "create", "one", "-f", "first", "--file", "second"],
        vec!["nss", "create", "one", "-f"],
        vec!["nss", "create", "one", "--file="],
        vec!["nss", "create", "one", "--file", "--dry-run"],
        vec!["nss", "create", "one", "--force"],
        vec!["nss", "create", "one", "--unknown"],
        vec!["nss", "create", "one", "--dry-run=true"],
        vec![
            "nss",
            "create",
            "one",
            "-od",
            "first",
            "--ollama-dir",
            "second",
        ],
        vec!["nss", "create", "../outside"],
        vec!["nss", "create", "ollama:one"],
        vec!["nss", "create", "one", "two"],
        vec!["nss", "create", "one", "--file=first", "-f=second"],
        vec!["nss", "create", "one", "--future-flag"],
    ] {
        let output = isolated_lma(&temp)
            .args(&argv)
            .arg("--dry-run")
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(2),
            "{argv:?}: {}",
            stderr(&output)
        );
        assert!(stdout(&output).is_empty());
        assert!(!stderr(&output).is_empty());
        assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
    }
}

#[test]
fn serve_dry_run_never_autodiscovers_a_cwd_modelfile_or_serializes_override_bookkeeping() {
    let temp = TempDir::new();
    let modelfile = temp.0.join("Modelfile");
    fs::write(&modelfile, "RUN do-not-execute-or-parse-this\n").unwrap();
    let output = isolated_lma(&temp)
        .env("HOME", temp.0.join("missing-home"))
        .args([
            "nss",
            "serve",
            "missing-model.gguf",
            "--context=4096",
            "--chat-template=auto",
            "--dry-run",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let json: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_eq!(json["command"], "serve");
    assert!(json["options"].get("definition").is_none());
    assert!(json["options"].get("explicit_load").is_none());
    assert!(stderr(&output).is_empty());
    assert_eq!(
        fs::read_to_string(modelfile).unwrap(),
        "RUN do-not-execute-or-parse-this\n"
    );
    assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 1);
}

#[test]
fn preview_is_embedded_configuration_not_a_shell_command() {
    let output = lma()
        .args([
            "nss",
            "serve",
            "missing model.gguf",
            "--gpu",
            "all",
            "--context",
            "8k",
            "--api-key",
            "secret-value",
            "--dry-run",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(!text.contains("secret-value"));
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(value["command"], "serve");
    assert_eq!(value["options"]["context"], 8192);
    assert_eq!(value["options"]["gpu_layers"], 999);
    assert_eq!(value["options"]["api_key"], "[redacted]");
}

#[test]
fn serve_gpu_default_is_feature_aware_and_explicit_cpu_always_wins() {
    let default_layers = if cfg!(any(
        feature = "metal",
        feature = "cuda",
        all(target_os = "macos", target_arch = "aarch64")
    )) {
        999
    } else {
        0
    };
    for prefix in [
        vec!["nss", "serve"],
        vec!["nss", "server"],
        vec!["nss", "serve"],
    ] {
        for fallback in ["error", "mmap"] {
            for (flags, expected) in [
                (vec![], default_layers),
                (vec!["--gpu", "cpu"], 0),
                (vec!["--gpu=0"], 0),
                (vec!["--n-gpu-layers", "0"], 0),
                (vec!["--gpu", "all", "--gpu", "cpu"], 0),
                (vec!["--gpu", "1"], 1),
            ] {
                let output = lma()
                    .args(&prefix)
                    .args(["--model", "missing.gguf", "--memory-fallback", fallback])
                    .args(flags)
                    .arg("--dry-run")
                    .output()
                    .unwrap();
                assert!(output.status.success(), "{}", stderr(&output));
                let json: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
                assert_eq!(json["options"]["gpu_layers"], expected);
                assert_eq!(json["options"]["memory_fallback"], fallback);
                assert!(!stderr(&output).contains("Retrying"));
            }
        }
    }
}

#[test]
fn serve_help_explains_backend_defaults_and_actual_placement() {
    let output = lma().args(["nss", "serve", "--help"]).output().unwrap();
    assert!(output.status.success());
    let help = stdout(&output);
    for expected in [
        "default: all in Metal/CUDA builds, else cpu",
        "actual placement",
        "not automatic memory fitting",
        "-g cpu",
    ] {
        assert!(help.contains(expected), "missing {expected}: {help}");
    }
}

#[test]
fn gpu_request_reports_runtime_availability_without_silent_fallback() {
    let temp = TempDir::new();
    let model = temp.0.join("header-only.gguf");
    // Enough for local header validation. With a GPU, the native loader must
    // preserve the malformed-model error; without one, availability fails first.
    fs::write(&model, b"GGUF\x03\x00\x00\x00").unwrap();
    let output = lma()
        .args(["nss", "serve"])
        .arg(&model)
        .args(["--gpu", "1"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let error = stderr(&output);
    for expected in [
        "Backend request: compiled=cpu",
        "requested=gpu",
        "effective gpu_layers=1; mode=Requested",
    ] {
        assert!(error.contains(expected), "missing {expected}: {error}");
    }
    if error.contains("GPU offload available=false") {
        for expected in [
            "GPU layers requested (1) but no GPU backend is available",
            "--features metal",
            "--gpu cpu",
        ] {
            assert!(error.contains(expected), "missing {expected}: {error}");
        }
    } else {
        assert!(error.contains("GPU offload available=true"), "{error}");
        assert!(error.contains("failed to load"), "{error}");
        assert!(error.contains("header-only.gguf"), "{error}");
        assert!(!error.contains("no GPU backend is available"), "{error}");
    }
    assert!(!error.contains("Retrying"));
    assert!(!error.contains("CPU/pageable mmap weights"));
}

#[test]
fn memory_and_cache_options_are_opt_in_without_changing_gpu_selection() {
    let output = lma()
        .args(["nss", "serve", "m.gguf", "--gpu", "all", "--dry-run"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let json: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_eq!(json["options"]["memory_fallback"], "error");
    assert_eq!(json["options"]["expert_cache_mib"], 0);
    assert_eq!(json["options"]["micro_batch"], 128);
    assert_eq!(json["options"]["kv_cache"], "f16");
    assert_eq!(json["options"]["prompt_cache"], true);
    let output = lma()
        .args([
            "nss",
            "serve",
            "-m",
            "m.gguf",
            "--gpu",
            "all",
            "--memory-fallback",
            "mmap",
            "--kv-cache",
            "q8_0",
            "--flash",
            "on",
            "--no-prompt-cache",
            "--dry-run",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let json: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_eq!(json["options"]["memory_fallback"], "mmap");
    assert_eq!(json["options"]["gpu_layers"], 999);
    assert_eq!(json["options"]["kv_cache"], "q8_0");
    assert_eq!(json["options"]["prompt_cache"], false);
    assert!(!stderr(&output).contains("WARNING"));
}

#[test]
fn expert_cache_dry_runs_share_host_settings_across_command_families() {
    let temp = TempDir::new();
    for prefix in [
        vec!["nss", "serve"],
        vec!["nss", "chat"],
        vec!["nss", "chat"],
        vec!["nss", "launch"],
        vec!["nss", "serve"],
        vec!["nss", "chat"],
        vec!["nss", "launch"],
        vec!["nss", "lh"],
        vec!["nss", "serve", "--"],
        vec!["nss", "chat", "--"],
        vec!["nss", "launch", "--"],
    ] {
        for (flags, budget, micro_batch, gpu_layers) in [
            (vec![], 0, 128, 999),
            (vec!["--expert-cache-mib", "1"], 1, 128, 999),
            (vec!["--expert-cache-mib=1048576"], 1_048_576, 128, 999),
            (
                vec!["--expert-cache-mib=64", "--batch-size=32"],
                64,
                32,
                999,
            ),
            (
                vec!["--batch-size=32", "--expert-cache-mib=64"],
                64,
                32,
                999,
            ),
            (vec!["--expert-cache-mib=64", "--ubatch-size=1"], 64, 1, 999),
            (vec!["--ubatch-size=1", "--expert-cache-mib=64"], 64, 1, 999),
            (
                vec!["--expert-cache-mib=64", "--n-gpu-layers=0"],
                64,
                128,
                0,
            ),
            (
                vec!["--n-gpu-layers=1", "--expert-cache-mib=64"],
                64,
                128,
                1,
            ),
            (vec!["--expert-cache-mib=64", "--ubatch-size=2"], 64, 2, 999),
            (
                vec!["--ubatch-size=32", "--expert-cache-mib=64"],
                64,
                32,
                999,
            ),
        ] {
            let output = isolated_lma(&temp)
                .stdin(std::process::Stdio::null())
                .env("HOME", temp.0.join("missing-home"))
                .env("OLLAMA_MODELS", temp.0.join("missing-store"))
                .args(&prefix)
                .args(["--path", "missing.gguf", "--n-gpu-layers=999"])
                .args(&flags)
                .arg("--dry-run")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{prefix:?} {flags:?}: {}",
                stderr(&output)
            );
            let json: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
            assert_eq!(json["dry_run"], true);
            assert_eq!(json["options"]["expert_cache_mib"], budget);
            assert_eq!(json["options"]["micro_batch"], micro_batch);
            assert_eq!(json["options"]["gpu_layers"], gpu_layers);
            assert!(json["options"].get("explicit_load").is_none());
            assert!(stderr(&output).is_empty(), "{}", stderr(&output));
            assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
        }
    }
}

#[test]
fn expert_cache_invalid_budgets_fail_before_loading_or_store_resolution() {
    let temp = TempDir::new();
    for prefix in [
        vec!["nss", "serve"],
        vec!["nss", "chat"],
        vec!["nss", "launch"],
        vec!["nss", "serve", "--"],
        vec!["nss", "chat", "--"],
        vec!["nss", "launch", "--"],
    ] {
        for flags in [
            vec!["--expert-cache-mib", "0"],
            vec!["--expert-cache-mib=0"],
            vec!["--expert-cache-mib=-1"],
            vec!["--expert-cache-mib=invalid"],
            vec!["--expert-cache-mib=1.5"],
            vec!["--expert-cache-mib=1k"],
            vec!["--expert-cache-mib=1048577"],
            vec!["--expert-cache-mib=4294967295"],
            vec!["--expert-cache-mib=4294967296"],
            vec!["--expert-cache-mib="],
            vec!["--expert-cache-mib"],
        ] {
            for dry_run in [false, true] {
                let mut command = isolated_lma(&temp);
                command
                    .args(&prefix)
                    .args([
                        "--ollama",
                        "missing:latest",
                        "--ollama-dir",
                        "missing-store",
                    ])
                    .args(&flags);
                if dry_run {
                    command.arg("--dry-run");
                }
                let output = command.output().unwrap();
                assert_eq!(
                    output.status.code(),
                    Some(2),
                    "{prefix:?} {flags:?}: {}",
                    stderr(&output)
                );
                assert!(stderr(&output).contains("--expert-cache-mib"));
                assert!(stdout(&output).is_empty());
                assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
            }
        }
    }
}

#[test]
fn expert_cache_rejects_projectors_in_either_order() {
    let temp = TempDir::new();
    for prefix in [
        vec!["nss", "serve"],
        vec!["nss", "chat"],
        vec!["nss", "launch"],
        vec!["nss", "serve"],
        vec!["nss", "chat"],
        vec!["nss", "launch"],
        vec!["nss", "serve", "--"],
        vec!["nss", "chat", "--"],
        vec!["nss", "launch", "--"],
    ] {
        for (flag, value, error) in [
            ("--projector", "missing-projector.gguf", "vision projector"),
            ("--mmproj", "missing-projector.gguf", "vision projector"),
        ] {
            if prefix.contains(&"--") && flag == "--micro-batch" {
                continue;
            }
            for cache_first in [false, true] {
                let cache = ["--expert-cache-mib", "64"];
                let incompatible = [flag, value];
                let (first, second) = if cache_first {
                    (cache, incompatible)
                } else {
                    (incompatible, cache)
                };
                let output = isolated_lma(&temp)
                    .args(&prefix)
                    .args(["--model", "missing.gguf"])
                    .args(first)
                    .args(second)
                    .arg("--dry-run")
                    .output()
                    .unwrap();
                assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
                assert!(stderr(&output).contains(error), "{}", stderr(&output));
                assert!(stdout(&output).is_empty());
                assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
            }
        }
    }
}

#[test]
fn expert_cache_help_documents_budget_backing_and_limits_without_a_model() {
    for prefix in [
        vec!["s", "-h"],
        vec!["c", "-h"],
        vec!["lh", "-h"],
        vec!["nss", "serve", "--help"],
        vec!["nss", "chat", "--help"],
        vec!["nss", "launch", "--help"],
    ] {
        let output = lma().env("PATH", "").args(&prefix).output().unwrap();
        assert!(output.status.success(), "{}", stderr(&output));
        let help = stdout(&output);
        for expected in [
            "-e N",
            "not total RSS",
            "mmap GGUF",
            "GPU-resident",
            "no projector",
            "requested batch/micro-batch",
            "2..16 local tokens",
            "-mh 16",
            "metadata/workspaces",
            "synchronous",
            "prefill",
            "Unsupported architectures error",
            "Modelfile",
        ] {
            assert!(
                help.contains(expected),
                "{prefix:?}: missing {expected}: {help}"
            );
        }
    }
}

#[test]
fn invalid_memory_and_cache_options_are_errors() {
    for flags in [
        vec!["--memory-fallback", "swap-on"],
        vec!["--kv-cache", "q8_0"],
        vec!["--kv-cache", "made-up"],
        vec!["--no-prompt-cache=true"],
    ] {
        let output = lma()
            .args(["nss", "serve", "m.gguf"])
            .args(flags)
            .arg("--dry-run")
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
    }
}

#[test]
fn ollama_preview_resolves_existing_blobs_and_projector_without_a_daemon() {
    let temp = TempDir::new();
    let manifest_dir = temp
        .0
        .join("manifests/registry.ollama.ai/library/vision-test");
    let blobs = temp.0.join("blobs");
    fs::create_dir_all(&manifest_dir).unwrap();
    fs::create_dir(&blobs).unwrap();
    let model_hash = "a".repeat(64);
    let projector_hash = "b".repeat(64);
    let model_path = blobs.join(format!("sha256-{model_hash}"));
    let projector_path = blobs.join(format!("sha256-{projector_hash}"));
    fs::write(&model_path, b"GGUFfixture").unwrap();
    fs::write(&projector_path, b"GGUFfixture").unwrap();
    let manifest = serde_json::json!({"schemaVersion":2,"layers":[
        {"mediaType":"application/vnd.ollama.image.model","digest":format!("sha256:{model_hash}")},
        {"mediaType":"application/vnd.ollama.image.projector","digest":format!("sha256:{projector_hash}")}
    ]});
    fs::write(manifest_dir.join("latest"), manifest.to_string()).unwrap();
    for (args, directory_flag) in [
        (
            vec!["nss", "serve", "--ollama", "vision-test"],
            "--ollama-dir",
        ),
        (vec!["nss", "serve", "-o", "vision-test"], "-od"),
        (
            vec!["nss", "serve", "-o=vision-test:latest"],
            "--ollama-dir",
        ),
        (vec!["nss", "serve", "-o", "vision-test:latest"], "-od"),
    ] {
        let output = lma()
            .env("PATH", "")
            .env("OLLAMA_MODELS", "/does-not-exist")
            .args(args)
            .arg(directory_flag)
            .arg(&temp.0)
            .arg("--dry-run")
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", stderr(&output));
        let json: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
        assert_eq!(json["options"]["model"], model_path.to_str().unwrap());
        assert_eq!(
            json["options"]["projector"],
            projector_path.to_str().unwrap()
        );
        assert_eq!(json["options"]["model_id"], "vision-test:latest");
        assert!(stderr(&output).contains("are not applied"));
    }
    let output = lma()
        .env("OLLAMA_MODELS", &temp.0)
        .args([
            "nss",
            "serve",
            "ollama:vision-test",
            "--mmproj",
            "override.gguf",
            "--dry-run",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let json: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_eq!(json["options"]["projector"], "override.gguf");
    assert_eq!(fs::read(model_path).unwrap(), b"GGUFfixture");
    assert_eq!(fs::read(projector_path).unwrap(), b"GGUFfixture");
}

#[test]
fn quantization_preview_creates_no_files() {
    let temp = TempDir::new();
    let output = lma()
        .current_dir(&temp.0)
        .args(["nss", "quantize", "in.gguf", "--type", "q8_0", "--dry-run"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let json: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_eq!(json["command"], "quantize");
    assert_eq!(json["options"]["output"], "in.Q8_0.gguf");
    assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
}

#[test]
fn fetch_dry_run_defaults_to_combined_catalogs_without_network_or_files() {
    let temp = TempDir::new();
    for argv in [
        vec!["nss", "fetch", "  owner/repository  ", "--dry-run"],
        vec!["f", "  owner/repository  ", "-d"],
    ] {
        let output = isolated_lma(&temp)
            .env("HOME", &temp.0)
            .env("USERPROFILE", &temp.0)
            .env("LMA_MODELS", "ignored-download-override")
            .args(&argv)
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", stderr(&output));
        let plan: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
        assert_eq!(
            plan,
            serde_json::json!({
                "command": "fetch",
                "dry_run": true,
                "options": {
                    "query": "owner/repository",
                    "ollama_database": true,
                    "lm_studio_database": true,
                    "list": false,
                    "file": null,
                    "output": temp.0.join(".lma/models")
                }
            })
        );
        assert!(stderr(&output).is_empty(), "{}", stderr(&output));
        assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
    }
}

#[test]
fn fetch_default_is_independent_of_cwd_and_missing_home_never_falls_back_to_cwd() {
    let home = TempDir::new();
    let first = TempDir::new();
    let second = TempDir::new();
    for cwd in [&first, &second] {
        let output = isolated_lma(cwd)
            .env_clear()
            .env("PATH", "")
            .env("HOME", &home.0)
            .env("USERPROFILE", &home.0)
            .args(["nss", "fetch", "owner/repo", "--dry-run"])
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", stderr(&output));
        let plan: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
        assert_eq!(
            plan["options"]["output"],
            serde_json::json!(home.0.join(".lma/models"))
        );
        for relative_home in [None, Some("relative-home")] {
            let mut command = isolated_lma(cwd);
            command.env_clear().env("PATH", "");
            if let Some(value) = relative_home {
                command.env("HOME", value).env("USERPROFILE", value);
            }
            let output = command
                .args(["nss", "fetch", "owner/repo", "--dry-run"])
                .output()
                .unwrap();
            assert_eq!(output.status.code(), Some(2));
            assert!(stderr(&output).contains("absolute home directory"));
        }
        let output = isolated_lma(cwd)
            .env_clear()
            .env("PATH", "")
            .args(["nss", "fetch", "owner/repo", "--out", ".", "--dry-run"])
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", stderr(&output));
        let plan: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
        assert_eq!(plan["options"]["output"], ".");
        assert_eq!(fs::read_dir(&cwd.0).unwrap().count(), 0);
    }
    assert_eq!(fs::read_dir(&home.0).unwrap().count(), 0);
}

#[test]
fn fetch_dry_run_honors_source_filters_and_selection_options() {
    let temp = TempDir::new();
    for (flags, ollama, lm_studio, file) in [
        (vec!["--ollama-database"], true, false, None),
        (
            vec!["--lm-studio-database", "--file", "weights/model.gguf"],
            false,
            true,
            Some("weights/model.gguf"),
        ),
        (
            vec!["--ollama-database", "--lm-studio-database"],
            true,
            true,
            None,
        ),
    ] {
        let output = isolated_lma(&temp)
            .args(["nss", "--dry-run", "fetch", "owner/repository"])
            .args(&flags)
            .args(["--list", "-o", "downloads"])
            .output()
            .unwrap();
        assert!(output.status.success(), "{flags:?}: {}", stderr(&output));
        let plan: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
        assert_eq!(plan["command"], "fetch");
        assert_eq!(plan["dry_run"], true);
        assert_eq!(plan["options"]["ollama_database"], ollama);
        assert_eq!(plan["options"]["lm_studio_database"], lm_studio);
        assert_eq!(plan["options"]["list"], true);
        assert_eq!(plan["options"]["file"], serde_json::json!(file));
        assert_eq!(plan["options"]["output"], "downloads");
        assert!(stderr(&output).is_empty(), "{}", stderr(&output));
        assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
    }
    let output = isolated_lma(&temp)
        .args([
            "nss",
            "fetch",
            "model",
            "--ollama-database",
            "--file",
            "model.gguf",
            "--dry-run",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("--lm-studio-database"));
    assert!(stdout(&output).is_empty());
    assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
}

#[test]
fn image_dry_run_needs_no_model_or_backend_and_creates_no_files() {
    let temp = TempDir::new();
    for argv in [
        vec!["nss", "image", "missing-sd15", "a red fox", "--dry-run"],
        vec!["nss", "imagen", "missing-sd15", "a red fox", "--dry-run"],
        vec!["nss", "image-gen", "missing-sd15", "a red fox", "--dry-run"],
        vec!["nss", "img", "missing-sd15", "a red fox", "--dry-run"],
        vec!["i", "missing-sd15", "a red fox", "-d"],
    ] {
        let output = isolated_lma(&temp).args(&argv).output().unwrap();
        assert!(output.status.success(), "{argv:?}: {}", stderr(&output));
        let plan: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
        assert_eq!(
            plan,
            serde_json::json!({
                "command": "image",
                "dry_run": true,
                "options": {
                    "model_dir": "missing-sd15",
                    "tokenizer": null,
                    "prompt": "a red fox",
                    "negative_prompt": "",
                    "output": "image.png",
                    "width": 512,
                    "height": 512,
                    "steps": 20,
                    "seed": 42,
                    "guidance": 7.5,
                    "device": "auto",
                    "attention_slicing": null
                }
            })
        );
        assert!(stderr(&output).is_empty(), "{}", stderr(&output));
        assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
    }
}

#[test]
fn nss_image_accepts_full_names_and_option_shortcuts() {
    let temp = TempDir::new();
    for command in ["image", "imagen", "image-gen", "img"] {
        let output = isolated_lma(&temp)
            .args([
                "nss",
                "--dry-run",
                command,
                "-m",
                "missing-sd15",
                "-p",
                "a red fox",
                "-n",
                "blurry",
                "-o",
                "result.png",
                "--tokenizer",
                "missing-tokenizer.json",
                "--width",
                "64",
                "--height",
                "1024",
                "--steps",
                "150",
                "--seed",
                "0",
                "--guidance",
                "1",
                "--device",
                "cpu",
                "--attention-slicing",
                "8",
            ])
            .output()
            .unwrap();
        assert!(output.status.success(), "{command}: {}", stderr(&output));
        let plan: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
        assert_eq!(plan["command"], "image");
        assert_eq!(plan["dry_run"], true);
        assert_eq!(
            plan["options"],
            serde_json::json!({
                "model_dir": "missing-sd15",
                "tokenizer": "missing-tokenizer.json",
                "prompt": "a red fox",
                "negative_prompt": "blurry",
                "output": "result.png",
                "width": 64,
                "height": 1024,
                "steps": 150,
                "seed": 0,
                "guidance": 1.0,
                "device": "cpu",
                "attention_slicing": 8
            })
        );
        assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
    }
}

#[test]
fn image_dry_run_still_validates_options() {
    let temp = TempDir::new();
    for (flags, message) in [
        (vec!["--width", "56"], "width and height"),
        (vec!["--width", "511"], "width and height"),
        (vec!["--height", "1032"], "width and height"),
        (vec!["--height", "65"], "width and height"),
        (vec!["--steps", "0"], "steps"),
        (vec!["--steps", "151"], "steps"),
        (vec!["--seed", "-1"], "seed"),
        (vec!["--guidance", "NaN"], "guidance"),
        (vec!["--guidance", "inf"], "guidance"),
        (vec!["--guidance", "0.5"], "guidance"),
        (vec!["--guidance", "21"], "guidance"),
        (vec!["--device", "unknown"], "device"),
        (vec!["--attention-slicing", "0"], "attention slicing"),
        (vec!["--attention-slicing", "3"], "attention slicing"),
        (
            vec!["--guidance", "1", "--attention-slicing", "16"],
            "positive divisor of 8",
        ),
        (vec!["-m", "other"], "model directory specified twice"),
        (vec!["-p", "other"], "prompt specified twice"),
        (vec!["extra"], "too many image arguments"),
    ] {
        let output = isolated_lma(&temp)
            .args([
                "nss",
                "image",
                "-m",
                "missing-sd15",
                "-p",
                "a red fox",
                "--dry-run",
            ])
            .args(&flags)
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(2),
            "{flags:?}: {}",
            stderr(&output)
        );
        assert!(
            stderr(&output).contains(message),
            "{flags:?}: {}",
            stderr(&output)
        );
        assert!(stdout(&output).is_empty());
        assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
    }
    for args in [
        vec![],
        vec!["missing-sd15"],
        vec!["", "prompt"],
        vec!["missing-sd15", "   "],
    ] {
        let output = isolated_lma(&temp)
            .args(["nss", "image"])
            .args(&args)
            .arg("--dry-run")
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(2),
            "{args:?}: {}",
            stderr(&output)
        );
        assert!(stdout(&output).is_empty());
    }
}

#[test]
fn nss_uses_supported_long_options() {
    let output = lma()
        .args([
            "nss",
            "--dry-run",
            "serve",
            "--model",
            "model.gguf",
            "--ctx-size",
            "8192",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let json: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_eq!(json["options"]["context"], 8192);
    let output = lma()
        .args(["nss", "quantize", "in", "out", "Q8_0", "4", "--dry-run"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let json: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_eq!(json["options"]["kind"], "Q8_0");
    assert_eq!(json["options"]["threads"], 4);
}

#[test]
fn nss_rejects_single_letter_commands_without_changing_normal_shortcuts() {
    for alias in ["s", "sp", "q", "f", "i", "it", "ce", "l", "c", "t", "h"] {
        for prefix in [vec!["nss"], vec!["nss", "--dry-run"]] {
            let output = lma().args(prefix).args([alias, "--help"]).output().unwrap();
            assert_eq!(output.status.code(), Some(2), "alias {alias} was accepted");
            assert!(stderr(&output).contains("root short commands are disabled"));
        }
        let output = lma().args([alias, "-h"]).output().unwrap();
        assert!(output.status.success(), "{}", stderr(&output));
    }
}

#[test]
fn nss_preserves_option_shortcuts_and_longer_command_aliases() {
    for command in ["serve", "server"] {
        for arguments in [vec!["-m", "model.gguf"], vec!["model.gguf"]] {
            let output = lma()
                .args(["nss", command])
                .args(arguments)
                .args(["-c", "8k", "-t", "2", "--gpu", "cpu", "--dry-run"])
                .output()
                .unwrap();
            assert!(output.status.success(), "{}", stderr(&output));
            let config: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
            assert_eq!(config["options"]["context"], 8192);
            assert_eq!(config["options"]["threads"], 2);
        }
    }
    for command in ["quantize", "quant"] {
        let output = lma()
            .args([
                "nss",
                command,
                "in.gguf",
                "-o",
                "out.gguf",
                "-q",
                "q8_0",
                "-t",
                "2",
                "--dry-run",
            ])
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", stderr(&output));
        let config: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
        assert_eq!(config["options"]["kind"], "Q8_0");
        assert_eq!(config["options"]["output"], "out.gguf");
    }
    for argv in [
        vec!["nss", "-h"],
        vec!["nss", "serve", "-h"],
        vec!["nss", "quantize", "-h"],
    ] {
        let output = lma().args(argv).output().unwrap();
        assert!(output.status.success(), "{}", stderr(&output));
    }
}

#[test]
fn executable_selection_and_unknown_features_are_rejected() {
    for argv in [
        vec!["nss", "serve", "m", "--backend", "program"],
        vec!["nss", "serve", "--backend-help"],
        vec!["nss", "serve", "m", "--", "--jinja"],
        vec!["nss", "serve", "--model", "m", "--future-option"],
        vec!["nss", "quantize", "in", "--importance", "matrix"],
    ] {
        let output = lma().args(argv).output().unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(stderr(&output).contains("unsupported"));
    }
}

#[test]
fn old_executable_environment_variables_are_not_used() {
    let output = lma()
        .env("LMA_SERVER", "do-not-run-me")
        .env("LMA_QUANTIZE", "do-not-run-me")
        .env("PATH", "")
        .args(["nss", "serve", "m", "--dry-run"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(!stdout(&output).contains("do-not-run-me"));
}

#[test]
fn quantization_refuses_missing_input_without_creating_output() {
    let temp = TempDir::new();
    let output = lma()
        .current_dir(&temp.0)
        .env("PATH", "")
        .args(["nss", "quantize", "missing.gguf"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("cannot access input"));
    assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
}

#[test]
fn quantization_never_overwrites_input_including_nss() {
    let temp = TempDir::new();
    let input = temp.0.join("model.gguf");
    fs::write(&input, "original model").unwrap();
    for command in [vec!["nss", "quantize"], vec!["nss", "quantize"]] {
        let output = lma()
            .args(command)
            .arg(&input)
            .arg(&input)
            .arg("--force")
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1));
        assert!(stderr(&output).contains("same file"));
        assert_eq!(fs::read_to_string(&input).unwrap(), "original model");
    }
}

#[test]
fn quantization_existing_output_requires_force() {
    let temp = TempDir::new();
    let input = temp.0.join("in.gguf");
    let output_path = temp.0.join("out.gguf");
    fs::write(&input, "input").unwrap();
    fs::write(&output_path, "original output").unwrap();
    let output = lma()
        .args(["nss", "quantize"])
        .arg(&input)
        .arg(&output_path)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("--force"));
    assert_eq!(fs::read_to_string(&output_path).unwrap(), "original output");
}

#[test]
fn root_help_is_a_local_versioned_command_tree() {
    let temp = TempDir::new();
    let output = isolated_lma(&temp)
        .env_clear()
        .env("PATH", "")
        .env("HOME", temp.0.join("missing-home"))
        .env("OLLAMA_MODELS", temp.0.join("missing-store"))
        .args(["-h"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.starts_with(concat!("lma ", env!("CARGO_PKG_VERSION"))));
    for expected in [
        "├─ h [COMMAND]",
        "├─ l [FILTER]",
        "├─ lh [NAME]",
        "├─ c [NAME]",
        "├─ t [OPTIONS]",
        "├─ s NAME",
        "├─ sp [-m NAME]",
        "│  ├─ s -p FILE",
        "│  └─ s -ct PRESET",
        "├─ q INPUT",
        "├─ f QUERY",
        "├─ it",
        "├─ ce NAME",
        "├─ i DIR PROMPT",
        "imagen",
        "└─ nss COMMAND",
        "lma nss help",
        "lma nss list",
        "lma nss launch",
        "lma nss chat",
        "lma nss transcribe",
        "lma nss stop",
        "lma COMMAND -h",
        "lma h s -a",
        "lma nss help serve advanced",
    ] {
        assert!(text.contains(expected), "missing {expected}: {text}");
    }
    assert!(text.lines().count() < 45, "root help should stay concise");
    assert!(stderr(&output).is_empty());
    assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
}

#[test]
fn command_help_describes_connection_and_imagegen_limits() {
    for command in ["serve", "launch", "chat"] {
        let output = lma()
            .env("PATH", "")
            .args(["nss", command, "--help"])
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", stderr(&output));
        let help = stdout(&output);
        for expected in ["-a ", "-ct ", "-d ", "-g ", "-or ", "-p "] {
            assert!(help.contains(expected), "{command}: missing {expected}");
        }
        for client in ["zed", "vscode", "opencode"] {
            assert!(help.contains(client), "{command}: missing {client}");
        }
    }
    let output = lma()
        .env("PATH", "")
        .args(["nss", "help", "serve", "advanced"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let help = stdout(&output);
    for expected in [
        "/v1/messages",
        "/v1/messages/count_tokens",
        "HTTP 200",
        "{\"input_tokens\":N}",
        "served GGUF tokenizer",
        "applied chat template",
        "generation special tokens",
        "not Anthropic's proprietary token count",
        "/v1/responses",
        "constrained/buffered",
        "native streaming",
        "client-executed",
        "no hosted shell",
        "not full",
        "Tools",
        "unsupported",
    ] {
        assert!(help.contains(expected), "HTTP help missing {expected}");
    }
    for obsolete in [
        "count_tokens returns HTTP 400",
        "exact token counting unsupported",
        "unsupported count_tokens endpoint",
    ] {
        assert!(!help.contains(obsolete), "obsolete HTTP help: {obsolete}");
    }
    let output = lma()
        .env("PATH", "")
        .args(["nss", "imagen", "--help"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let help = stdout(&output);
    for expected in [
        "SD1.5",
        "source builds",
        "No automatic model downloads",
        "not GGUF, SDXL, Flux",
        "-a N",
        "Unavailable: Candle 0.11 tensor-rank bug",
        "rejects any value before loading weights",
        "-d can still preview config",
        "omit this flag for generation",
    ] {
        assert!(help.contains(expected), "image help missing {expected}");
    }
}

#[test]
fn terminal_reply_caps_are_configurable_without_changing_host_api_limits() {
    let temp = TempDir::new();
    for argv in [
        vec!["c", "model", "-ts", "off", "-c", "8k", "-d"],
        vec!["lh", "model", "-ts", "0", "-d"],
        vec!["nss", "chat", "model", "--tokens", "off", "--dry-run"],
    ] {
        let output = isolated_lma(&temp).args(argv).output().unwrap();
        assert!(output.status.success(), "{}", stderr(&output));
        let plan: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(plan["options"]["chat_tokens"], 0);
    }
    let output = isolated_lma(&temp)
        .args(["c", "model", "-ts", "2048", "-c", "8k", "-d"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let plan: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(plan["options"]["chat_tokens"], 2048);
    for argv in [
        vec!["c", "model", "-ts", "4096", "-d"],
        vec!["c", "model", "-ts", "nope", "-d"],
        vec!["c", "model", "-ts=-1", "-d"],
        vec!["c", "model", "--tokens", "off", "-d"],
        vec!["s", "model", "-ts", "off", "-d"],
        vec!["nss", "serve", "model", "--tokens", "off", "--dry-run"],
    ] {
        let output = isolated_lma(&temp).args(&argv).output().unwrap();
        assert_eq!(output.status.code(), Some(2), "{argv:?}");
        assert!(output.stdout.is_empty());
    }
    assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
}

#[test]
fn terminal_help_documents_final_keys_slashes_and_attachment_limits() {
    let output = lma()
        .env("PATH", "")
        .args(["nss", "launch", "--help"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let help = stdout(&output);
    for expected in [
        "Up/Down",
        "Tab/Shift+Tab",
        "Left/Right",
        "Enter edits",
        "Esc cancels an edit or quits setup",
        "lma h c",
    ] {
        assert!(help.contains(expected), "launch help missing {expected}");
    }
    let output = lma()
        .env("PATH", "")
        .args(["nss", "chat", "--help"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let help = stdout(&output);
    for expected in [
        "Enter / Ctrl+Enter",
        "Shift+Enter",
        "Alt+Enter",
        "Newline fallback",
        "reported distinctly by the terminal",
        "Keyboard enhancement",
        "only on terminals supporting it",
        "Mouse wheel",
        "PgUp/PgDn",
        "Up / Down",
        "history is focused",
        "Home / End",
        "MARKDOWN DISPLAY",
        "headings, bold, italic",
        "inline/fenced code, lists, quotes, and links",
        "no commands are executed",
        "no URLs are opened",
        "Tab",
        "Esc",
        "Ctrl+C",
        "Cancel while generating; quit when idle",
        "Ctrl+Q",
        "PageUp / PageDown",
        "/help",
        "/new",
        "/models",
        "Unload model, clear conversation",
        "/system TEXT",
        "omit TEXT to clear",
        "/tokens N|off",
        "0 < N < context size",
        "-ts N|off",
        "off/0 removes the reply cap",
        "MODEL STAYS",
        "no automatic history truncation",
        "/plain",
        "/markdown",
        "/temperature N",
        "/raw",
        "/chat",
        "/image FILE",
        "Requires a projector",
        "max 8 images, 8 MiB/image",
        "16 MiB total",
        "24 MiB serialized JSON request",
        "/save FILE",
        "completed text and attachment names",
        "Existing files are never overwritten",
        "/logs",
        "/quit",
    ] {
        assert!(help.contains(expected), "chat help missing {expected}");
    }
}

#[test]
fn client_presets_default_api_but_explicit_api_wins_in_either_order() {
    for prefix in [
        vec!["nss", "serve"],
        vec!["nss", "launch"],
        vec!["nss", "chat"],
        vec!["nss", "serve"],
    ] {
        for (client, expected_api) in [
            ("default", "all"),
            ("claude-code", "anthropic"),
            ("codex", "openai"),
            ("zed", "openai"),
            ("vscode", "openai"),
            ("opencode", "openai"),
            ("custom", "all"),
        ] {
            let output = lma()
                .args(&prefix)
                .args(["missing.gguf", "--client", client, "--dry-run"])
                .output()
                .unwrap();
            assert!(output.status.success(), "{}", stderr(&output));
            let json: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
            assert_eq!(json["options"]["client"], client);
            assert_eq!(json["options"]["api"], expected_api);
            for api in ["all", "openai", "ollama", "anthropic"] {
                for flags in [
                    vec!["--api", api, "--client", client],
                    vec!["--client", client, "--api", api],
                ] {
                    let output = lma()
                        .args(&prefix)
                        .args(["missing.gguf", "--dry-run"])
                        .args(flags)
                        .output()
                        .unwrap();
                    assert!(output.status.success(), "{}", stderr(&output));
                    let json: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
                    assert_eq!(json["options"]["api"], api);
                    assert_eq!(json["options"]["client"], client);
                }
            }
        }
    }
}

#[test]
fn printed_client_profiles_escape_config_hide_auth_and_gate_disabled_apis() {
    use lma::{config::ServeConfig, runtime::clients::setup};
    let model = "org/model\"\n\\path\t\u{1b}\u{7f}";
    for client in ["claude-code", "codex", "zed", "vscode", "opencode"] {
        for authenticated in [false, true] {
            let mut config = ServeConfig {
                client: client.into(),
                api: if client == "claude-code" {
                    "anthropic"
                } else {
                    "openai"
                }
                .into(),
                model_id: Some(model.into()),
                api_key: authenticated.then(|| "do-not-print-this-key".into()),
                ..ServeConfig::default()
            };
            let address = "[::]:8099".parse().unwrap();
            let output = setup(&config, address);
            assert!(output.contains("http://[::1]:8099"));
            assert!(!output.contains("do-not-print-this-key"));
            assert!(!output.contains(model));
            assert!(!output.contains('\u{1b}'));
            assert!(!output.contains('\u{7f}'));
            let json_blocks: Vec<serde_json::Value> = output
                .split("```json\n")
                .skip(1)
                .map(|part| serde_json::from_str(part.split("\n```").next().unwrap()).unwrap())
                .collect();
            match client {
                "zed" => {
                    assert_eq!(
                        json_blocks[0]["language_models"]["openai_compatible"]["lma"]["available_models"]
                            [0]["name"],
                        model
                    );
                    assert_eq!(
                        json_blocks[1]["LMA_API_KEY"],
                        if authenticated {
                            "REPLACE_WITH_YOUR_LMA_API_KEY"
                        } else {
                            "local-not-required"
                        }
                    );
                }
                "opencode" => {
                    assert_eq!(
                        json_blocks[0]["provider"]["lma"]["npm"],
                        "@ai-sdk/openai-compatible"
                    );
                    assert_eq!(json_blocks[0]["model"], format!("lma/{model}"));
                    assert_eq!(json_blocks[0]["small_model"], json_blocks[0]["model"]);
                }
                "claude-code" => {
                    assert_eq!(json_blocks[0]["ANTHROPIC_MODEL"], model);
                    assert_eq!(json_blocks[0]["ANTHROPIC_DEFAULT_SONNET_MODEL"], model);
                    assert_eq!(
                        json_blocks[0]["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"],
                        "1"
                    );
                }
                "codex" | "vscode" => {
                    let prefix = if client == "codex" {
                        "model = "
                    } else {
                        "    model: "
                    };
                    let encoded = output
                        .lines()
                        .find_map(|line| line.strip_prefix(prefix))
                        .unwrap();
                    assert_eq!(serde_json::from_str::<String>(encoded).unwrap(), model);
                    if client == "codex" {
                        assert!(output.contains("supports_websockets = false"));
                        assert!(output.contains("enable_request_compression = false"));
                        assert!(output.contains("user-level ~/.codex/config.toml"));
                    } else {
                        assert!(output.contains("Continue extension"));
                        assert!(output.contains("useResponsesApi: false"));
                    }
                }
                _ => unreachable!(),
            }
            assert!(output.contains("end-to-end compatibility is not verified"));
            config.api = "ollama".into();
            let gated = setup(&config, address);
            assert!(gated.contains("no usable client profile emitted"));
            assert!(!gated.contains("```"));
            assert!(!gated.contains("do-not-print-this-key"));
        }
    }
}

#[test]
fn new_client_presets_keep_nss_shortcuts_inline_values_and_auth_redaction() {
    let temp = TempDir::new();
    for client in ["zed", "vscode", "opencode"] {
        let output = isolated_lma(&temp)
            .args([
                "nss",
                "serve",
                "-m",
                "missing.gguf",
                "-c",
                "8k",
                "--api-key",
                "secret-value",
                "--dry-run",
            ])
            .arg(format!("--client={client}"))
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", stderr(&output));
        let json: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
        assert_eq!(json["options"]["client"], client);
        assert_eq!(json["options"]["api"], "openai");
        assert_eq!(json["options"]["context"], 8192);
        assert_eq!(json["options"]["api_key"], "[redacted]");
        assert!(!stdout(&output).contains("secret-value"));
        assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
    }
    let output = lma()
        .args([
            "nss",
            "serve",
            "m.gguf",
            "--client",
            "continue",
            "--dry-run",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("--client must be one of:"));
}

#[test]
fn launch_and_chat_dry_runs_need_no_tty_or_model_and_do_not_resolve_stores() {
    let temp = TempDir::new();
    for prefix in [
        vec!["nss", "launch"],
        vec!["nss", "launch"],
        vec!["nss", "lh"],
        vec!["nss", "chat"],
        vec!["nss", "--dry-run", "launch"],
        vec!["nss", "chat"],
    ] {
        let expected_mode = if prefix.contains(&"chat") {
            "chat"
        } else {
            "launch"
        };
        let output = isolated_lma(&temp)
            .stdin(std::process::Stdio::null())
            .env("OLLAMA_MODELS", temp.0.join("does-not-exist"))
            .args(prefix)
            .args([
                "--ollama-dir",
                "missing-store",
                "--api-key",
                "secret",
                "--dry-run",
            ])
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", stderr(&output));
        let json: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
        assert_eq!(json["command"], "launch");
        assert_eq!(json["dry_run"], true);
        assert_eq!(json["options"]["model"], "");
        assert_eq!(json["options"]["mode"], expected_mode);
        assert_eq!(json["options"]["target"], "chat");
        assert_eq!(json["options"]["interactive"], true);
        assert_eq!(json["options"]["api"], "all");
        assert_eq!(json["options"]["client"], "default");
        assert_eq!(json["options"]["ollama_dir"], "missing-store");
        assert_eq!(json["options"]["api_key"], "[redacted]");
        assert!(
            json["options"].get("serve").is_none(),
            "serve config must be flattened"
        );
        assert!(!stdout(&output).contains("secret"));
        assert!(stderr(&output).is_empty(), "{}", stderr(&output));
        assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
    }
}

#[test]
fn launch_shares_serve_options_and_preserves_model_refs_without_loading() {
    let temp = TempDir::new();
    for prefix in [
        vec!["nss", "launch"],
        vec!["nss", "launch"],
        vec!["nss", "launch"],
        vec!["nss", "lh"],
        vec!["nss", "chat"],
    ] {
        let expected_mode = if prefix.contains(&"chat") {
            "chat"
        } else {
            "launch"
        };
        for interactive in [false, true] {
            let mut command = isolated_lma(&temp);
            command.args(&prefix).args([
                "-o",
                "llama3.2:latest",
                "-od",
                "missing-store",
                "--gpu",
                "cpu",
                "-c",
                "8k",
                "-t",
                "2",
                "--batch",
                "64",
                "--flash",
                "on",
                "--kv-cache",
                "q8_0",
                "--dry-run",
            ]);
            if interactive {
                command.arg("--interactive");
            }
            let output = command.output().unwrap();
            assert!(output.status.success(), "{}", stderr(&output));
            let json: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
            let options = &json["options"];
            assert_eq!(options["model"], "ollama:llama3.2:latest");
            assert_eq!(options["interactive"], interactive);
            assert_eq!(options["mode"], expected_mode);
            assert_eq!(options["target"], "chat");
            assert_eq!(options["gpu_layers"], 0);
            assert_eq!(options["context"], 8192);
            assert_eq!(options["threads"], 2);
            assert_eq!(options["batch"], 64);
            assert_eq!(options["micro_batch"], 64);
            assert_eq!(options["kv_cache"], "q8_0");
        }
    }
    let output = isolated_lma(&temp)
        .args(["nss", "launch", "--client=codex", "--dry-run"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let json: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_eq!(json["options"]["mode"], "launch");
    assert_eq!(json["options"]["target"], "chat");
    assert_eq!(json["options"]["interactive"], true);
    assert_eq!(json["options"]["api"], "openai");
    assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
}

#[test]
fn launch_target_dry_runs_serialize_all_choices_without_starting_clients_or_models() {
    let temp = TempDir::new();
    for prefix in [vec!["nss", "launch"], vec!["nss", "launch"]] {
        for target in [
            "chat",
            "opencode",
            "codex",
            "claude",
            "opencode-desktop",
            "codex-desktop",
            "claude-desktop",
        ] {
            for model in [None, Some("tiny:latest")] {
                let mut command = isolated_lma(&temp);
                command
                    .stdin(std::process::Stdio::null())
                    .env("HOME", temp.0.join("missing-home"))
                    .env("OLLAMA_MODELS", temp.0.join("missing-store"))
                    .args(&prefix);
                if let Some(model) = model {
                    command.arg(model);
                }
                let output = command
                    .args([
                        "--target",
                        target,
                        "--client=claude-code",
                        "--api=ollama",
                        "--api-key=secret",
                        "--dry-run",
                    ])
                    .output()
                    .unwrap();
                assert!(
                    output.status.success(),
                    "{prefix:?} {target}: {}",
                    stderr(&output)
                );
                let json: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
                let options = &json["options"];
                assert_eq!(json["command"], "launch");
                assert_eq!(json["dry_run"], true);
                assert_eq!(options["mode"], "launch");
                assert_eq!(options["target"], target);
                assert_eq!(options["model"], model.unwrap_or(""));
                assert_eq!(options["interactive"], model.is_none());
                assert_eq!(options["client"], "claude-code");
                assert_eq!(options["api"], "ollama");
                assert_eq!(options["api_key"], "[redacted]");
                assert!(!stdout(&output).contains("secret"));
                assert!(stderr(&output).is_empty(), "{}", stderr(&output));
                assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
            }
        }
    }
    let output = isolated_lma(&temp)
        .args([
            "nss",
            "launch",
            "tiny",
            "--target=codex-desktop",
            "--dry-run",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let json: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_eq!(json["options"]["target"], "codex-desktop");
    assert_eq!(json["options"]["client"], "default");
    assert_eq!(json["options"]["api"], "all");
    assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
}

#[test]
fn launch_target_errors_reject_non_launch_commands_and_invalid_choices() {
    for argv in [
        vec!["nss", "launch", "--target"],
        vec!["nss", "launch", "--target="],
        vec!["nss", "launch", "--target=desktop"],
        vec!["nss", "launch", "--target=CODEX"],
        vec!["nss", "chat", "--target=chat"],
        vec!["nss", "chat", "--target=opencode"],
        vec!["nss", "chat", "--target=codex"],
        vec!["nss", "serve", "tiny", "--target=claude"],
        vec!["nss", "serve", "tiny", "--target=chat"],
        vec!["nss", "launch", "one", "two", "--target=codex"],
    ] {
        let output = lma()
            .env("PATH", "")
            .args(&argv)
            .arg("--dry-run")
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(2),
            "{argv:?}: {}",
            stdout(&output)
        );
        assert!(stdout(&output).is_empty());
        assert!(!stderr(&output).is_empty());
    }
}

#[test]
fn launch_target_help_documents_choices_desktop_gates_and_long_nss_examples() {
    let output = lma()
        .env("PATH", "")
        .args(["nss", "launch", "--help"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let help = stdout(&output);
    for expected in [
        "-tt CHOICE",
        "chat (default)",
        "opencode",
        "codex",
        "claude",
        "opencode-desktop",
        "codex-desktop",
        "claude-desktop",
        "gated",
        "-ct remains a print-only",
        "typed target",
        "desktop app opened",
        "lma nss launch llama3.2 --target codex",
    ] {
        assert!(help.contains(expected), "missing {expected}: {help}");
    }
    assert!(!help.contains("no external agent processes"));
    assert!(!help.contains("lma nss lh"));
}

#[test]
fn serve_still_requires_a_model_and_invalid_launch_profiles_are_explicit_errors() {
    for prefix in [
        vec!["nss", "serve"],
        vec!["nss", "server"],
        vec!["nss", "serve"],
    ] {
        let output = lma().args(prefix).arg("--dry-run").output().unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(stderr(&output).contains("serve needs an installed model NAME or file path"));
    }
    for command in ["serve", "launch", "chat"] {
        for (flag, value, expected) in [
            (
                "--api",
                "bogus",
                "--api must be all, openai, ollama, or anthropic",
            ),
            (
                "--client",
                "bogus",
                "--client must be one of: default, claude-code, codex, zed, vscode, opencode, custom",
            ),
        ] {
            let output = lma()
                .args(["nss", command, "m.gguf", flag, value, "--dry-run"])
                .output()
                .unwrap();
            assert_eq!(output.status.code(), Some(2));
            assert!(stderr(&output).contains(expected), "{}", stderr(&output));
        }
    }
    for argv in [
        vec!["nss", "chat", "--mode", "host", "--dry-run"],
        vec!["nss", "chat", "--mode=host", "--dry-run"],
        vec!["nss", "launch", "--mode", "agent", "--dry-run"],
        vec!["nss", "launch", "m", "--context", "1", "--dry-run"],
        vec!["nss", "launch", "m", "--kv-cache", "q8_0", "--dry-run"],
        vec!["nss", "launch", "--interactive=yes", "--dry-run"],
        vec!["nss", "launch", "m", "--", "-o", "llama3.2", "--dry-run"],
        vec!["nss", "serve", "m", "--mode", "chat", "--dry-run"],
    ] {
        let output = lma().args(&argv).output().unwrap();
        assert_eq!(
            output.status.code(),
            Some(2),
            "{argv:?}: {}",
            stdout(&output)
        );
        if argv.iter().any(|arg| arg.starts_with("--mode")) {
            assert!(stderr(&output).contains("--mode has been removed"));
        }
    }
}

#[test]
fn list_and_l_discover_local_and_ollama_models_without_a_daemon() {
    let temp = TempDir::new();
    let local = temp.0.join("models");
    let extra = temp.0.join("extra");
    let store = temp.0.join("ollama");
    let manifests = store.join("manifests/registry.ollama.ai/library/test-model");
    fs::create_dir_all(&local).unwrap();
    fs::create_dir_all(&extra).unwrap();
    fs::create_dir_all(&manifests).unwrap();
    fs::create_dir_all(store.join("blobs")).unwrap();
    fs::write(local.join("local.gguf"), b"GGUFfixture").unwrap();
    fs::write(extra.join("extra.gguf"), b"GGUFfixture").unwrap();
    fs::write(local.join("ignored.gguf.part"), b"GGUFfixture").unwrap();
    let hash = "a".repeat(64);
    fs::write(store.join(format!("blobs/sha256-{hash}")), b"GGUFfixture").unwrap();
    let manifest = serde_json::json!({"schemaVersion":2,"layers":[{
        "mediaType":"application/vnd.ollama.image.model", "digest":format!("sha256:{hash}")
    }]});
    fs::write(manifests.join("latest"), manifest.to_string()).unwrap();
    for prefix in [
        vec!["nss", "list"],
        vec!["nss", "list"],
        vec!["nss", "list"],
    ] {
        let output = isolated_lma(&temp)
            .env_clear()
            .env("PATH", "")
            .env("HOME", temp.0.join("missing-home"))
            .env("USERPROFILE", temp.0.join("missing-home"))
            .env("OLLAMA_MODELS", &store)
            .env("LMA_MODELS", &extra)
            .args(prefix)
            .arg("--json")
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", stderr(&output));
        let catalog: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
        let models = catalog["models"].as_array().unwrap();
        assert_eq!(models.len(), 3, "{catalog}");
        assert!(
            models
                .iter()
                .any(|m| m["name"] == "local.gguf" && m["ollama"] == false)
        );
        assert!(models.iter().any(|m| m["name"] == "extra.gguf"));
        let ollama = models.iter().find(|m| m["ollama"] == true).unwrap();
        assert_eq!(ollama["model"], "test-model:latest");
        assert_eq!(ollama["sizeBytes"], 11);
        assert!(catalog["warnings"].as_array().unwrap().is_empty());
        assert!(stderr(&output).is_empty());
    }
    let output = isolated_lma(&temp)
        .env_clear()
        .env("PATH", "")
        .args(["nss", "list", "TEST-MODEL", "-od"])
        .arg(&store)
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    for heading in ["NAME", "SOURCE", "SIZE", "MODEL"] {
        assert!(text.contains(heading));
    }
    assert!(text.contains("test-model:latest"));
    assert!(!text.contains("local.gguf"));
    assert_eq!(fs::read(local.join("local.gguf")).unwrap(), b"GGUFfixture");
    assert_eq!(
        fs::read(manifests.join("latest")).unwrap(),
        manifest.to_string().as_bytes()
    );
}

#[test]
fn list_empty_json_dry_run_help_and_nss_are_consistent() {
    let temp = TempDir::new();
    let output = isolated_lma(&temp)
        .env_clear()
        .env("PATH", "")
        .args(["nss", "list", "--json", "-od", "missing-store"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let empty: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_eq!(empty, serde_json::json!({"models":[], "warnings":[]}));
    let output = isolated_lma(&temp)
        .env_clear()
        .env("PATH", "")
        .args([
            "nss",
            "list",
            "--dry-run",
            "--dir=missing-local",
            "-od=missing-store",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let plan: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_eq!(plan["command"], "list");
    assert_eq!(plan["dry_run"], true);
    assert_eq!(
        plan["options"]["local_dirs"],
        serde_json::json!(["missing-local"])
    );
    for argv in [
        vec!["nss", "list", "-h"],
        vec!["nss", "list", "--help"],
        vec!["nss", "help", "l"],
        vec!["nss", "list", "-h"],
    ] {
        let output = isolated_lma(&temp).args(argv).output().unwrap();
        assert!(output.status.success(), "{}", stderr(&output));
        assert!(stdout(&output).contains("lma l —"));
    }
    let output = isolated_lma(&temp).args(["nss", "l"]).output().unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("root short commands are disabled"));
    assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
}

#[test]
fn chat_template_overrides_work_across_cli_entrypoints_and_nss() {
    for prefix in [
        vec!["nss", "serve"],
        vec!["nss", "launch"],
        vec!["nss", "chat"],
        vec!["nss", "serve"],
        vec!["nss", "lh"],
    ] {
        for template in [
            "auto",
            "native",
            "plain",
            "chatml",
            "llama3",
            "gemma",
            "mistral-v7",
            "gpt-oss",
        ] {
            let output = lma()
                .env("PATH", "")
                .args(&prefix)
                .args([
                    "-m",
                    "not-loaded.gguf",
                    "--chat-template",
                    template,
                    "--dry-run",
                ])
                .output()
                .unwrap();
            assert!(output.status.success(), "{}", stderr(&output));
            let plan: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
            assert_eq!(plan["options"]["chat_template"], template);
        }
    }
    let output = lma()
        .args(["nss", "serve", "not-loaded.gguf", "--dry-run"])
        .output()
        .unwrap();
    let plan: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_eq!(plan["options"]["chat_template"], "auto");
    for flags in [vec!["--chat-template"], vec!["--chat-template=made-up"]] {
        let output = lma()
            .args(["nss", "serve", "m.gguf"])
            .args(flags)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(stderr(&output).contains("--chat-template"));
    }
}

#[test]
fn import_requires_explicit_output_and_reads_only_the_selected_store_on_dry_run() {
    let temp = TempDir::new();
    let output = isolated_lma(&temp)
        .args(["nss", "import", "-o", "gemma4:12b"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("--out"));
    let output = isolated_lma(&temp)
        .env("OLLAMA_MODELS", "unrelated-store")
        .args([
            "nss",
            "import",
            "-o",
            "gemma4:12b",
            "-od",
            "missing-store",
            "--out",
            "new.gguf",
            "--dry-run",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("cannot access Ollama model store"));
    assert!(stdout(&output).is_empty());
    assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
}

#[test]
fn delete_requires_confirmation_and_source_guard_and_never_deletes_external_alias_weights() {
    use lma::models::{registry, registry::Entry};
    let temp = TempDir::new();
    let home = temp.0.join("home");
    fs::create_dir(&home).unwrap();
    let root = home.join(".lma/models");
    let external = temp.0.join("external.gguf");
    fs::write(&external, b"GGUFexternal").unwrap();
    registry::put(
        &root,
        "shared",
        Entry {
            path: external.clone(),
            projector: None,
        },
    )
    .unwrap();
    let store = home.join("ollama-store");
    let manifest = store.join("manifests/registry.ollama.ai/library/shared/latest");
    let blob = store
        .join("blobs")
        .join(format!("sha256-{}", "a".repeat(64)));
    fs::create_dir_all(manifest.parent().unwrap()).unwrap();
    fs::create_dir_all(blob.parent().unwrap()).unwrap();
    fs::write(&blob, b"GGUFollama").unwrap();
    fs::write(&manifest, serde_json::to_vec(&serde_json::json!({"schemaVersion":2,"layers":[{
        "mediaType":"application/vnd.ollama.image.model","digest":format!("sha256:{}", "a".repeat(64)),"size":10
    }]})).unwrap()).unwrap();
    let run = |args: &[&str]| {
        isolated_lma(&temp)
            .stdin(std::process::Stdio::null())
            .env("HOME", &home)
            .env("USERPROFILE", &home)
            .env("LMA_MODELS", home.join("missing-models"))
            .env("OLLAMA_MODELS", &store)
            .args(args)
            .output()
            .unwrap()
    };
    let before = fs::read(root.join(".registry-v1.json")).unwrap();
    for args in [
        vec!["d", "shared", "-s", "lma", "-d"],
        vec!["nss", "delete", "shared", "--source", "lma", "--dry-run"],
    ] {
        let preview = run(&args);
        assert!(preview.status.success(), "{}", stderr(&preview));
        let plan: serde_json::Value = serde_json::from_str(&stdout(&preview)).unwrap();
        assert_eq!(plan["command"], "delete");
        assert_eq!(plan["options"]["model"], "shared");
        assert_eq!(plan["options"]["yes"], false);
        assert_eq!(fs::read(root.join(".registry-v1.json")).unwrap(), before);
    }
    let refused = run(&["d", "shared", "-s", "lma"]);
    assert_eq!(refused.status.code(), Some(1));
    assert!(
        stderr(&refused).contains("confirmation"),
        "{}",
        stderr(&refused)
    );
    assert_eq!(fs::read(root.join(".registry-v1.json")).unwrap(), before);
    let ambiguous = run(&["d", "shared", "-y"]);
    assert_eq!(ambiguous.status.code(), Some(1));
    assert!(
        stderr(&ambiguous).contains("ambiguous"),
        "{}",
        stderr(&ambiguous)
    );
    assert!(manifest.exists());
    let deleted = run(&["d", "shared", "-s", "lma", "-y"]);
    assert!(deleted.status.success(), "{}", stderr(&deleted));
    assert!(!registry::read(&root).unwrap().contains_key("shared:latest"));
    assert_eq!(fs::read(&external).unwrap(), b"GGUFexternal");
    assert!(manifest.exists() && blob.exists());
    let deleted = run(&["nss", "delete", "shared", "--source", "ollama", "--yes"]);
    assert!(deleted.status.success(), "{}", stderr(&deleted));
    assert!(!manifest.exists());
    assert!(!blob.exists());
    assert_eq!(fs::read(&external).unwrap(), b"GGUFexternal");
}

#[test]
fn delete_managed_weights_only_after_the_last_alias_is_removed() {
    use lma::models::{registry, registry::Entry};
    let temp = TempDir::new();
    let home = temp.0.join("home");
    let root = home.join(".lma/models");
    fs::create_dir_all(&root).unwrap();
    let weights = root.join("managed.gguf");
    fs::write(&weights, b"GGUFmanaged").unwrap();
    for label in ["first", "second"] {
        registry::put(
            &root,
            label,
            Entry {
                path: weights.clone(),
                projector: None,
            },
        )
        .unwrap();
    }
    for (label, remains) in [("first", true), ("second", false)] {
        let output = isolated_lma(&temp)
            .stdin(std::process::Stdio::null())
            .env("HOME", &home)
            .env("USERPROFILE", &home)
            .env("LMA_MODELS", home.join("missing-models"))
            .env("OLLAMA_MODELS", home.join("missing-ollama"))
            .args(["d", label, "-s", "lma", "-y"])
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", stderr(&output));
        assert_eq!(weights.exists(), remains);
    }
    assert!(registry::read(&root).unwrap().is_empty());
}

#[test]
fn application_process_launching_is_confined_to_the_explicit_launch_module() {
    fn check(path: &std::path::Path, launcher: &std::path::Path) {
        for entry in fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                check(&path, launcher);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let source = fs::read_to_string(&path).unwrap();
                for forbidden in ["CommandExt", "std::process::exit", ".exec()"] {
                    assert!(
                        !source.contains(forbidden),
                        "{} contains {forbidden}",
                        path.display()
                    );
                }
                for forbidden in ["Command::new", "process::Command"] {
                    if path == launcher {
                        continue;
                    }
                    assert!(
                        !source.contains(forbidden),
                        "{} contains {forbidden}",
                        path.display()
                    );
                }
            }
        }
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    check(&root, &root.join("runtime/launch.rs"));
}
