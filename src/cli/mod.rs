mod args;
mod create;
mod delete;
mod fetch;
pub(crate) mod help;
mod image;
mod import;
mod list;
mod nss;
mod quantize;
mod serve;
mod stop;
mod transcribe;
mod update;

use crate::config::Plan;
use args::{Args, short_arguments};
use std::ffi::{OsStr, OsString};

pub enum Action {
    Help(&'static str),
    Version,
    Run(Box<Plan>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Syntax {
    Friendly,
    Nss,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Command {
    Serve,
    Stop,
    Launch,
    List,
    Chat,
    Quantize,
    Fetch,
    Image,
    Import,
    Create,
    Delete,
    Transcribe,
    Update,
}

pub fn parse(arguments: impl IntoIterator<Item = OsString>) -> Result<Action, String> {
    let mut args = Args::new(arguments);
    let Some(command) = args.next() else {
        return Ok(Action::Help(help::ROOT));
    };
    if let Some(command) = command_named(&command) {
        return parse_command(command, args, Syntax::Friendly);
    }
    match command.to_str() {
        Some("nss") => nss::parse(args),
        Some("h") => help::parse(args, Syntax::Friendly),
        Some("-v") if args.is_empty() => Ok(Action::Version),
        Some("-h") if args.is_empty() => Ok(Action::Help(help::ROOT)),
        Some(name) if nss_command_named(&command).is_some() || name == "help" => Err(format!(
            "'{name}' is an NSS command; use `lma nss {name} ...` or `lma nss help {name}`; see `lma h` for canonical root commands"
        )),
        Some("--help" | "--help-all" | "--version" | "-V" | "--") => Err(format!(
            "'{}' is not supported at root; use `lma -h` / `lma -v`, or `lma nss --help` / `lma nss --version`; see `lma nss help` for long syntax",
            command.to_string_lossy()
        )),
        _ => Err(format!(
            "unknown command '{}'; try `lma h`",
            command.to_string_lossy()
        )),
    }
}

fn command_named(command: &OsStr) -> Option<Command> {
    match command.to_str()? {
        "s" => Some(Command::Serve),
        "sp" => Some(Command::Stop),
        "lh" => Some(Command::Launch),
        "l" => Some(Command::List),
        "c" => Some(Command::Chat),
        "q" => Some(Command::Quantize),
        "f" => Some(Command::Fetch),
        "it" => Some(Command::Import),
        "ce" => Some(Command::Create),
        "d" => Some(Command::Delete),
        "t" => Some(Command::Transcribe),
        "i" => Some(Command::Image),
        "u" => Some(Command::Update),
        _ => None,
    }
}

pub(super) fn nss_command_named(command: &OsStr) -> Option<Command> {
    match command.to_str()? {
        "serve" | "server" => Some(Command::Serve),
        "stop" => Some(Command::Stop),
        "launch" | "lh" => Some(Command::Launch),
        "list" => Some(Command::List),
        "chat" => Some(Command::Chat),
        "quantize" | "quant" => Some(Command::Quantize),
        "fetch" => Some(Command::Fetch),
        "image" | "image-gen" | "imagen" | "img" => Some(Command::Image),
        "import" => Some(Command::Import),
        "create" => Some(Command::Create),
        "delete" => Some(Command::Delete),
        "transcribe" => Some(Command::Transcribe),
        "update" => Some(Command::Update),
        _ => None,
    }
}

fn parse_command(command: Command, args: Args, syntax: Syntax) -> Result<Action, String> {
    let args = match syntax {
        Syntax::Friendly => short_arguments(command, args)?,
        Syntax::Nss => args,
    };
    match command {
        Command::Serve | Command::Launch | Command::Chat => serve::parse(args, command),
        Command::Stop => stop::parse(args),
        Command::Quantize => quantize::parse(args, syntax),
        Command::Fetch => fetch::parse(args),
        Command::List => list::parse(args),
        Command::Image => image::parse(args),
        Command::Import => import::parse(args),
        Command::Create => create::parse(args),
        Command::Delete => delete::parse(args),
        Command::Transcribe => transcribe::parse(args),
        Command::Update => update::parse(args),
    }
}

fn unsupported(option: &str) -> String {
    format!(
        "unsupported option '{option}': lma now runs an embedded engine, not an external llama.cpp command. Run `lma h` for supported features"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn plan(argv: &[&str]) -> Plan {
        match parse(argv.iter().map(OsString::from)).unwrap() {
            Action::Run(plan) => *plan,
            _ => panic!("expected a run plan"),
        }
    }

    #[test]
    fn empty_invocation_shows_help() {
        assert!(matches!(parse([]).unwrap(), Action::Help(_)));
    }

    const ROOT_COMMANDS: &[&str] = &[
        "s", "sp", "lh", "l", "c", "q", "f", "i", "it", "ce", "d", "t", "u",
    ];

    #[test]
    fn canonical_root_commands_route_directly_to_the_correct_jobs() {
        use crate::config::Job;
        for (argv, expected) in [
            (vec!["s", "-m", "model.gguf", "-d"], "serve"),
            (vec!["sp", "-m", "local:latest", "-d"], "stop"),
            (vec!["u", "-c", "-d"], "update"),
            (vec!["lh", "-i", "-d"], "launch"),
            (vec!["c", "-i", "-d"], "chat"),
            (vec!["l", "-j", "-dr", "local", "-o", "store", "-d"], "list"),
            (
                vec!["q", "in.gguf", "-t", "q8_0", "-ts", "4", "-d"],
                "quantize",
            ),
            (vec!["f", "model", "-l", "-o", "downloads", "-d"], "fetch"),
            (
                vec!["i", "-m", "sd15", "-p", "a fox", "-de", "cpu", "-d"],
                "image",
            ),
            (
                vec!["it", "-o", "local:latest", "-ot", "out.gguf", "-d"],
                "import",
            ),
            (
                vec!["ce", "local:latest", "-f", "missing-Modelfile", "-d"],
                "create",
            ),
            (vec!["d", "local:latest", "-s", "lma", "-d"], "delete"),
            (
                vec!["t", "-m", "whisper", "-a", "missing.wav", "-g", "off", "-d"],
                "transcribe",
            ),
        ] {
            let parsed = plan(&argv);
            assert!(parsed.dry_run, "{argv:?}");
            let actual = match parsed.job {
                Job::Serve(_) => "serve",
                Job::Stop(_) => "stop",
                Job::Launch(config) => {
                    assert!(config.interactive);
                    assert_eq!(config.mode, expected);
                    if config.mode == "launch" {
                        "launch"
                    } else {
                        "chat"
                    }
                }
                Job::List(_) => "list",
                Job::Quantize(config) => {
                    assert_eq!(config.kind, "Q8_0");
                    assert_eq!(config.threads, 4);
                    "quantize"
                }
                Job::Fetch(_) => "fetch",
                Job::Image(_) => "image",
                Job::Import(_) => "import",
                Job::Create(_) => "create",
                Job::Delete(_) => "delete",
                Job::Transcribe(_) => "transcribe",
                Job::Update(_) => "update",
            };
            assert_eq!(actual, expected, "{argv:?}");
        }
    }

    #[test]
    fn all_old_long_commands_are_nss_only_including_legacy_aliases() {
        for command in [
            "serve",
            "server",
            "stop",
            "launch",
            "list",
            "chat",
            "quantize",
            "quant",
            "fetch",
            "image",
            "image-gen",
            "imagen",
            "img",
            "import",
            "create",
            "transcribe",
            "help",
        ] {
            let Err(error) = parse([command.into(), "--help".into()]) else {
                panic!("root accepted {command}");
            };
            assert!(error.contains(&format!("lma nss {command}")), "{error}");
            assert!(error.contains("lma nss help"), "{error}");
            assert!(
                matches!(
                    parse(["nss".into(), command.into(), "--help".into()]).unwrap(),
                    Action::Help(_)
                ),
                "NSS rejected {command}"
            );
        }
        assert!(matches!(
            parse(["nss".into(), "lh".into(), "--help".into()]).unwrap(),
            Action::Help(_)
        ));
    }

    #[test]
    fn root_long_flags_and_separator_are_rejected_before_help_for_every_family() {
        for command in ROOT_COMMANDS {
            for option in [
                "--",
                "--help",
                "--help-all",
                "--version",
                "--dry-run",
                "--model=x",
                "--future",
                "-V",
                "-od=store",
                "-unknown",
            ] {
                for flags in [vec![option], vec!["-h", option], vec![option, "-h"]] {
                    let mut argv = vec![*command];
                    argv.extend(flags);
                    let Err(error) = parse(argv.iter().map(OsString::from)) else {
                        panic!("accepted {argv:?}");
                    };
                    assert!(error.contains("lma nss"), "{argv:?}: {error}");
                }
            }
            assert!(matches!(
                parse([(*command).into(), "-h".into()]).unwrap(),
                Action::Help(_)
            ));
        }
    }

    #[test]
    fn global_help_and_version_are_short_only_and_nss_keeps_old_flags() {
        for argv in [
            vec!["h"],
            vec!["-h"],
            vec!["nss"],
            vec!["nss", "--help"],
            vec!["nss", "-h"],
            vec!["nss", "help"],
        ] {
            assert!(matches!(
                parse(argv.into_iter().map(OsString::from)).unwrap(),
                Action::Help(_)
            ));
        }
        for argv in [
            vec!["-v"],
            vec!["nss", "--version"],
            vec!["nss", "-V"],
            vec!["nss", "-v"],
        ] {
            assert!(matches!(
                parse(argv.into_iter().map(OsString::from)).unwrap(),
                Action::Version
            ));
        }
        for argv in [
            vec!["--help"],
            vec!["--version"],
            vec!["-V"],
            vec!["--"],
            vec!["-h", "--help"],
            vec!["-v", "extra"],
            vec!["nss", "--version", "extra"],
        ] {
            assert!(parse(argv.into_iter().map(OsString::from)).is_err());
        }
    }

    #[test]
    fn canonical_flags_replace_old_aliases_and_values_are_not_options() {
        use crate::config::Job;
        let Job::Serve(config) = plan(&[
            "s",
            "-m=model--help.gguf",
            "-or=store--version",
            "-c=8k",
            "-mh=32",
            "-g=cpu",
            "-d",
        ])
        .job
        else {
            panic!()
        };
        assert_eq!(config.model, std::path::PathBuf::from("model--help.gguf"));
        assert_eq!(config.ollama_dir, Some("store--version".into()));
        assert_eq!(config.context, 8192);
        assert_eq!(config.micro_batch, 32);
        assert_eq!(config.gpu_layers, 0);
        let Job::Image(config) = plan(&[
            "i",
            "-m",
            "sd15",
            "-p",
            "text --help and --version are values",
            "-n=--negative",
            "-ht",
            "256",
            "-sd",
            "7",
            "-d",
        ])
        .job
        else {
            panic!()
        };
        assert_eq!(config.prompt, "text --help and --version are values");
        assert_eq!(config.negative_prompt, "--negative");
        assert_eq!(config.height, 256);
        assert_eq!(config.seed, 7);
        for argv in [
            vec!["q", "in.gguf", "-q", "q8_0"],
            vec!["s", "-m", "model", "-od", "store"],
            vec!["s", "-m", "model", "-V"],
            vec!["s", "-m", "model", "--", "--port", "9000"],
            vec!["q", "in.gguf", "-threads", "4"],
            vec!["s", "-h=true"],
            vec!["l", "-d=true"],
            vec!["q", "in.gguf", "-t"],
            vec!["s", "-m="],
            vec!["s", "-m", "--help"],
        ] {
            assert!(parse(argv.iter().map(OsString::from)).is_err(), "{argv:?}");
        }
        assert!(matches!(
            parse(["nss", "serve", "-m", "model", "-od", "store"].map(OsString::from)).unwrap(),
            Action::Run(_)
        ));
        assert!(matches!(
            parse(["nss", "quantize", "in.gguf", "-q", "q8_0"].map(OsString::from)).unwrap(),
            Action::Run(_)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn direct_parse_preserves_non_utf8_paths_in_separate_and_inline_values() {
        use crate::config::Job;
        use std::os::unix::ffi::OsStringExt;
        let path = OsString::from_vec(b"model--help-\xff.gguf".to_vec());
        for inline in [false, true] {
            let mut argv = vec!["s".into()];
            if inline {
                let mut argument = OsString::from("-m=");
                argument.push(&path);
                argv.push(argument);
            } else {
                argv.extend(["-m".into(), path.clone()]);
            }
            argv.push("-d".into());
            let Action::Run(plan) = parse(argv).unwrap() else {
                panic!()
            };
            let Job::Serve(config) = plan.job else {
                panic!()
            };
            assert_eq!(config.model.as_os_str(), path);
        }
        assert!(
            parse([
                "i".into(),
                "-m=sd15".into(),
                OsString::from_vec(b"-p=bad-\xff".to_vec())
            ])
            .is_err()
        );
        assert!(parse(["s".into(), OsString::from_vec(b"-\xff".to_vec())]).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn direct_parse_preserves_non_utf8_quantize_output_and_serve_path() {
        use crate::config::Job;
        use std::{os::unix::ffi::OsStringExt, path::PathBuf};

        // Absolute paths avoid serve's intentional './' prefix for relative -p paths.
        let path = OsString::from_vec(b"/nonexistent/model--help-\xff.gguf".to_vec());
        for (command, flag) in [("q", "-o"), ("s", "-p")] {
            for inline in [false, true] {
                let mut argv = vec![OsString::from(command)];
                if command == "q" {
                    argv.push("input.gguf".into());
                }
                if inline {
                    let mut argument = OsString::from(flag);
                    argument.push("=");
                    argument.push(&path);
                    argv.push(argument);
                } else {
                    argv.extend([flag.into(), path.clone()]);
                }
                argv.push("-d".into());
                let Action::Run(plan) = parse(argv.clone()).unwrap() else {
                    panic!("expected a run plan for {argv:?}");
                };
                assert!(plan.dry_run);
                match plan.job {
                    Job::Quantize(config) => {
                        assert_eq!(command, "q");
                        assert_eq!(config.input, PathBuf::from("input.gguf"));
                        assert_eq!(config.output, PathBuf::from(&path));
                    }
                    Job::Serve(config) => {
                        assert_eq!(command, "s");
                        assert_eq!(config.model, PathBuf::from(&path));
                    }
                    _ => panic!("incorrect job for {argv:?}"),
                }

                // Values containing '--' remain values, but later root long flags
                // must still be rejected before any parser can return help.
                argv.extend(["-h".into(), "--help".into()]);
                let Err(error) = parse(argv) else {
                    panic!("accepted a root long flag after a non-UTF-8 value");
                };
                assert!(error.contains("lma nss"), "{error}");
            }
        }
    }

    #[test]
    fn unsupported_commands_are_rejected() {
        for argv in [vec!["native"], vec!["unknown"], vec!["--version", "extra"]] {
            assert!(parse(argv.into_iter().map(OsString::from)).is_err());
        }
    }
}
