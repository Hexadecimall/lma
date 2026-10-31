use std::{collections::HashSet, path::PathBuf};

use super::{
    Action,
    args::{Args, Flag},
};
use crate::config::{DeleteConfig, Job, Plan};

pub const HELP: &str = "lma d — delete one installed model

USAGE
  lma d MODEL [-s SOURCE] [-o OLLAMA_DIR] [-y] [-d] [-h]
  lma d -m MODEL [-s SOURCE] [-o OLLAMA_DIR] [-y] [-d]

ROOT OPTIONS (canonical short flags only)
  -m MODEL        Model label/ID, instead of the single positional MODEL
  -s SOURCE       Source guard: lma, ollama, or lm-studio
  -o OLLAMA_DIR   Ollama store directory (not a source selector)
  -y              Skip confirmation; required when stdin is not a terminal
  -d              Preview arguments only; do not resolve or delete anything
  -h              Show this help

NSS EXAMPLES
  lma nss delete MODEL --source lma --dry-run
  lma nss delete --model=MODEL --source=ollama --ollama-dir=STORE --yes
  lma nss --dry-run delete -m MODEL --source lm-studio
  lma nss delete --help-all

NSS OPTIONS
  --model/-m, --source, --ollama-dir, --yes, --dry-run,
  --help/-h, and --help-all. Values support FLAG=VALUE, including values
  starting with '-'. Supply MODEL once, either positionally or with -m.
  Long commands/options are NSS-only; use lma h d or lma nss help delete.

SAFETY AND OUTPUT
  Unload the selected model first with lma sp -m MODEL (NSS: lma nss stop
  --model MODEL), or in the desktop app, and stop its use in other processes.
  This CLI cannot prove whether another process still has the model loaded.
  It never unloads models, contacts a host, or stops other processes for you.
  Deletion can permanently remove model files. Without -y/--yes, a terminal
  confirmation on stdin is required; only y or yes confirms (case-insensitive).
  Labels and discovered IDs (including source-prefixed IDs) are passed unchanged
  to the existing deletion API. Use -s/--source to disambiguate sources; a store
  directory alone does not select a source. Ambiguous matches are refused.
  A registry alias may be removed without deleting shared or external weights.
  Dry-run prints the Plan: an argument preview, NOT a resolved file list or a
  guarantee that the selected model exists or is deletable. Success prints the
  existing deletion result as pretty JSON. Nothing is downloaded or queried
  over the network.
  ~/.lma/models is lma's default download/registry location, not an ownership
  claim over Ollama, LM Studio, or external files referenced by an alias.
";

pub(super) fn parse(mut args: Args) -> Result<Action, String> {
    let mut model = None;
    let mut source = None;
    let mut ollama_dir = None;
    let mut yes = false;
    let mut dry_run = false;
    let mut help = false;
    let mut seen = HashSet::new();

    while let Some(argument) = args.next() {
        let Some(flag) = Flag::parse(&argument) else {
            let value = argument
                .into_string()
                .map_err(|_| "delete model label/ID must be UTF-8")?;
            validate_text(&value, "model label/ID")?;
            if model.replace(value).is_some() {
                return Err(
                    "delete accepts exactly one model, positionally or with -m/--model, not both"
                        .into(),
                );
            }
            continue;
        };
        let name = match flag.name.as_str() {
            "--model" | "-m" => "model",
            "--source" => "source",
            "--ollama-dir" => "ollama-dir",
            "--yes" => "yes",
            "--dry-run" => "dry-run",
            "--help" | "--help-all" | "-h" => "help",
            _ => {
                return Err(format!(
                    "unknown delete option '{}'; use `lma h d` or `lma nss help delete`",
                    flag.name
                ));
            }
        };
        if !seen.insert(name) {
            return Err(format!("delete {name} specified twice"));
        }
        match name {
            "model" => {
                let value = args.text(&flag)?;
                validate_text(&value, "model label/ID")?;
                if model.replace(value).is_some() {
                    return Err("delete accepts exactly one model, positionally or with -m/--model, not both".into());
                }
            }
            "source" => {
                let value = args.text(&flag)?;
                if !["lma", "ollama", "lm-studio"].contains(&value.as_str()) {
                    return Err("delete source must be lma, ollama, or lm-studio".into());
                }
                source = Some(value);
            }
            "ollama-dir" => {
                let value = args.value(&flag)?;
                if value.as_encoded_bytes().len() > 4096
                    || value.to_string_lossy().chars().any(char::is_control)
                {
                    return Err(
                        "delete Ollama directory must be 1..4096 bytes without control characters"
                            .into(),
                    );
                }
                ollama_dir = Some(PathBuf::from(value));
            }
            "yes" => {
                flag.switch()?;
                yes = true;
            }
            "dry-run" => {
                flag.switch()?;
                dry_run = true;
            }
            "help" => {
                flag.switch()?;
                help = true;
            }
            _ => unreachable!(),
        }
    }
    if help {
        return Ok(Action::Help(HELP));
    }
    let model = model.ok_or("delete needs one model label/ID: `lma d MODEL -d` or `lma nss delete --model MODEL --dry-run`")?;
    Ok(Action::Run(Box::new(Plan {
        job: Job::Delete(DeleteConfig {
            model,
            source,
            ollama_dir,
            yes,
        }),
        dry_run,
    })))
}

fn validate_text(value: &str, name: &str) -> Result<(), String> {
    if value.is_empty() || value.len() > 4096 || value.chars().any(char::is_control) {
        return Err(format!(
            "delete {name} must be 1..4096 bytes without control characters"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::tests::plan;
    use std::ffi::OsString;

    fn rejects(argv: &[&str]) {
        assert!(
            crate::cli::parse(argv.iter().map(OsString::from)).is_err(),
            "accepted {argv:?}"
        );
    }

    #[test]
    fn root_and_nss_build_the_same_argument_only_plan() {
        for argv in [
            vec![
                "d",
                "example:latest",
                "-s",
                "ollama",
                "-o",
                "missing-store",
                "-y",
                "-d",
            ],
            vec![
                "d",
                "-m=example:latest",
                "-s=ollama",
                "-o=missing-store",
                "-y",
                "-d",
            ],
            vec![
                "nss",
                "delete",
                "example:latest",
                "--source",
                "ollama",
                "--ollama-dir",
                "missing-store",
                "--yes",
                "--dry-run",
            ],
            vec![
                "nss",
                "--dry-run",
                "delete",
                "--model=example:latest",
                "--source=ollama",
                "--ollama-dir=missing-store",
                "--yes",
            ],
        ] {
            let parsed = plan(&argv);
            assert!(parsed.dry_run);
            let Job::Delete(config) = parsed.job else {
                panic!()
            };
            assert_eq!(config.model, "example:latest");
            assert_eq!(config.source.as_deref(), Some("ollama"));
            assert_eq!(config.ollama_dir, Some("missing-store".into()));
            assert!(config.yes);
        }
    }

    #[test]
    fn defaults_and_prefixed_ids_are_preserved_without_model_resolution() {
        for selector in [
            "example:tag",
            "lma:alias:latest",
            "ollama:example:latest",
            "lma:/missing/model.gguf",
            "ollama:5:store:example:tag",
        ] {
            for argv in [vec!["d", selector], vec!["nss", "delete", "-m", selector]] {
                let parsed = plan(&argv);
                assert!(!parsed.dry_run);
                let Job::Delete(config) = parsed.job else {
                    panic!()
                };
                assert_eq!(config.model, selector);
                assert!(config.source.is_none());
                assert!(config.ollama_dir.is_none());
                assert!(!config.yes);
            }
        }
        for source in ["lma", "ollama", "lm-studio"] {
            let Job::Delete(config) = plan(&["d", "example", "-s", source]).job else {
                panic!()
            };
            assert_eq!(config.source.as_deref(), Some(source));
        }
        let Job::Delete(config) = plan(&["d", "-m=-example", "-o=-store"]).job else {
            panic!()
        };
        assert_eq!(config.model, "-example");
        assert_eq!(config.ollama_dir, Some("-store".into()));
    }

    #[test]
    fn missing_invalid_and_duplicate_arguments_are_rejected() {
        for flags in [
            vec![],
            vec![""],
            vec!["one", "two"],
            vec!["--model"],
            vec!["--model="],
            vec!["--model", "--yes"],
            vec!["one", "--model=two"],
            vec!["-m=one", "two"],
            vec!["-m=one", "--model=one"],
            vec!["one", "--source"],
            vec!["one", "--source="],
            vec!["one", "--source=studio"],
            vec!["one", "--source=LMA"],
            vec!["one", "--source=lma", "--source=ollama"],
            vec!["one", "--ollama-dir"],
            vec!["one", "--ollama-dir="],
            vec!["one", "--ollama-dir", "--yes"],
            vec!["one", "--ollama-dir=a", "--ollama-dir=b"],
            vec!["one", "--yes", "--yes"],
            vec!["one", "--yes=true"],
            vec!["one", "--dry-run", "--dry-run"],
            vec!["one", "--dry-run=false"],
            vec!["--help", "--help-all"],
            vec!["--help=true"],
            vec!["--help", "--unknown"],
            vec!["--help", "--source"],
            vec!["one", "--"],
            vec!["one", "--mod=two"],
            vec!["-mone"],
            vec!["one", "--source=lma\n"],
            vec!["bad\0label"],
            vec!["bad\nlabel"],
            vec!["one", "--ollama-dir=bad\tdir"],
        ] {
            let mut argv = vec!["nss", "delete"];
            argv.extend(flags);
            rejects(&argv);
        }
        for flags in [
            vec!["one", "-m=two"],
            vec!["one", "-s=lma", "-s=lma"],
            vec!["one", "-o=a", "-o=b"],
            vec!["one", "-y", "-y"],
            vec!["one", "-d", "-d"],
            vec!["-h", "-h"],
            vec!["one", "--source=lma"],
            vec!["one", "-so=lma"],
        ] {
            let mut argv = vec!["d"];
            argv.extend(flags);
            rejects(&argv);
        }
    }

    #[test]
    fn model_and_directory_limits_are_in_bytes() {
        let maximum = "x".repeat(4096);
        let Job::Delete(config) = plan(&["d", &maximum, "-o", &maximum, "-d"]).job else {
            panic!()
        };
        assert_eq!(config.model, maximum);
        let too_long = "x".repeat(4097);
        rejects(&["d", &too_long]);
        rejects(&["d", "model", "-o", &too_long]);
        rejects(&["nss", "delete", "-m", &"é".repeat(2049)]);
    }

    #[test]
    fn help_explains_confirmation_unloading_and_argument_only_preview() {
        for argv in [
            vec!["d", "-h"],
            vec!["nss", "delete", "--help"],
            vec!["nss", "delete", "--help-all"],
            vec!["nss", "delete", "-h"],
        ] {
            let Action::Help(text) =
                crate::cli::parse(argv.into_iter().map(OsString::from)).unwrap()
            else {
                panic!()
            };
            assert_eq!(text, HELP);
            for detail in [
                "lma sp -m MODEL",
                "desktop",
                "argument preview",
                "NOT a resolved file list",
                "--yes",
                "--source",
                "--ollama-dir",
            ] {
                assert!(text.contains(detail), "missing {detail}");
            }
        }
        for argv in [
            vec!["delete", "model"],
            vec!["rm", "model"],
            vec!["nss", "rm", "model"],
        ] {
            rejects(&argv);
        }
    }

    #[cfg(unix)]
    #[test]
    fn directory_preserves_non_utf8_while_model_requires_utf8() {
        use std::os::unix::ffi::OsStringExt;
        let directory = OsString::from_vec(b"missing-store-\xff".to_vec());
        for inline in [false, true] {
            let mut argv = vec!["nss".into(), "delete".into(), "model".into()];
            if inline {
                let mut flag = OsString::from("--ollama-dir=");
                flag.push(&directory);
                argv.push(flag);
            } else {
                argv.extend(["--ollama-dir".into(), directory.clone()]);
            }
            let Action::Run(parsed) = crate::cli::parse(argv).unwrap() else {
                panic!()
            };
            let Job::Delete(config) = parsed.job else {
                panic!()
            };
            assert_eq!(config.ollama_dir.unwrap().as_os_str(), directory);
        }
        assert!(crate::cli::parse(["nss".into(), "delete".into(), directory]).is_err());
    }
}
