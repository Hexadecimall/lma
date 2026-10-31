use super::{
    Action,
    args::{Args, Flag},
    help,
};
use crate::config::{CreateConfig, Job, Plan};
use std::path::PathBuf;

pub(super) fn parse(mut args: Args) -> Result<Action, String> {
    let mut model = None;
    let mut file = None;
    let mut ollama_dir = None;
    let mut dry_run = false;
    while let Some(argument) = args.next() {
        if let Some(flag) = Flag::parse(&argument) {
            match flag.name.as_str() {
                "--help" | "--help-all" | "-h" => {
                    flag.switch()?;
                    return Ok(Action::Help(help::CREATE));
                }
                "--file" | "-f" => {
                    if file.replace(PathBuf::from(args.value(&flag)?)).is_some() {
                        return Err("create file specified twice; use one -f/--file PATH".into());
                    }
                }
                "--ollama-dir" | "-od" => {
                    if ollama_dir
                        .replace(PathBuf::from(args.value(&flag)?))
                        .is_some()
                    {
                        return Err("create Ollama directory specified twice".into());
                    }
                }
                "--dry-run" => {
                    flag.switch()?;
                    dry_run = true;
                }
                _ => {
                    return Err(format!(
                        "unknown create option '{}'; use `lma h ce`",
                        flag.name
                    ));
                }
            }
        } else if model
            .replace(
                argument
                    .into_string()
                    .map_err(|_| "create model NAME must be UTF-8")?,
            )
            .is_some()
        {
            return Err("create accepts exactly one model NAME".into());
        }
    }
    let model =
        model.ok_or("create needs a model NAME: `lma ce NAME -f Modelfile -d` or `lma nss create NAME --file Modelfile --dry-run`")?;
    // Validate alias syntax without touching stores or changing the supplied name.
    crate::models::registry::name(&model)?;
    Ok(Action::Run(Box::new(Plan {
        job: Job::Create(CreateConfig {
            model,
            file: file.unwrap_or_else(|| PathBuf::from("Modelfile")),
            ollama_dir,
        }),
        dry_run,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::tests::plan;
    use std::ffi::OsString;

    #[test]
    fn create_keeps_explicit_file_and_store_paths_including_nss_shortcuts() {
        for argv in [
            vec![
                "ce",
                "my-model:tag",
                "-f",
                "recipes/My Model",
                "-o",
                "missing-store",
                "-d",
            ],
            vec![
                "nss",
                "create",
                "my-model:tag",
                "--file=recipes/My Model",
                "--ollama-dir=missing-store",
                "--dry-run",
            ],
            vec![
                "nss",
                "create",
                "my-model:tag",
                "--file",
                "recipes/My Model",
                "-od=missing-store",
                "--dry-run",
            ],
            vec![
                "nss",
                "--dry-run",
                "create",
                "-f=recipes/My Model",
                "my-model:tag",
                "--ollama-dir",
                "missing-store",
            ],
        ] {
            let parsed = plan(&argv);
            assert!(parsed.dry_run);
            let Job::Create(config) = parsed.job else {
                panic!()
            };
            assert_eq!(config.model, "my-model:tag");
            assert_eq!(config.file, PathBuf::from("recipes/My Model"));
            assert_eq!(config.ollama_dir, Some(PathBuf::from("missing-store")));
        }
    }

    #[test]
    fn create_defaults_to_cwd_modelfile_without_reading_it_during_cli_parsing() {
        for argv in [vec!["ce", "my-model"], vec!["nss", "create", "my-model"]] {
            let parsed = plan(&argv);
            assert!(!parsed.dry_run);
            let Job::Create(config) = parsed.job else {
                panic!()
            };
            assert_eq!(config.model, "my-model");
            assert_eq!(config.file, PathBuf::from("Modelfile"));
            assert!(config.ollama_dir.is_none());
        }
        let parsed = plan(&["ce", "my-model", "-f", "missing-Modelfile", "-d"]);
        let json = serde_json::to_value(parsed).unwrap();
        assert_eq!(json["command"], "create");
        assert_eq!(json["dry_run"], true);
        assert_eq!(json["options"]["model"], "my-model");
        assert_eq!(json["options"]["file"], "missing-Modelfile");
    }

    #[test]
    fn create_rejects_duplicate_missing_and_unknown_arguments_before_io() {
        for argv in [
            vec!["ce"],
            vec!["ce", ""],
            vec!["ce", "one", "two"],
            vec!["ce", "one", "-f", "first", "-f", "second"],
            vec!["ce", "one", "-f"],
            vec!["ce", "../outside"],
            vec!["ce", "ollama:one"],
            vec!["nss", "create", "one", "-f", "first", "--file", "second"],
            vec!["nss", "create", "one", "--file=first", "-f=second"],
            vec!["nss", "create", "one", "--file="],
            vec!["nss", "create", "one", "--file", "--dry-run"],
            vec!["nss", "create", "one", "--ollama-dir="],
            vec![
                "nss",
                "create",
                "one",
                "-od",
                "first",
                "--ollama-dir",
                "second",
            ],
            vec!["nss", "create", "one", "--unknown"],
            vec!["nss", "create", "one", "--force"],
            vec!["nss", "create", "one", "--dry-run=true"],
            vec!["nss", "create", "one", "--help=yes"],
            vec!["nss", "create", "one", "two"],
            vec!["nss", "create", "one", "-f=first", "--file=second"],
            vec!["nss", "create", "one", "--future-flag"],
        ] {
            assert!(
                crate::cli::parse(argv.iter().map(OsString::from)).is_err(),
                "{argv:?}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn create_preserves_non_utf8_file_paths_but_requires_a_utf8_model_name() {
        use std::os::unix::ffi::OsStringExt;
        let path = OsString::from_vec(vec![b'm', 0xff]);
        let parsed = crate::cli::parse([
            OsString::from("nss"),
            OsString::from("create"),
            OsString::from("my-model"),
            OsString::from("-f"),
            path.clone(),
        ])
        .unwrap();
        let Action::Run(parsed) = parsed else {
            panic!()
        };
        let Job::Create(config) = parsed.job else {
            panic!()
        };
        assert_eq!(config.file.as_os_str(), path.as_os_str());
        assert!(crate::cli::parse([OsString::from("ce"), path]).is_err());
    }
}
