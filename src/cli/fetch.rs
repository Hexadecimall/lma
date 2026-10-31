use std::path::PathBuf;

use super::{
    Action,
    args::{Args, Flag},
    help,
};
use crate::config::{FetchConfig, Job, Plan};

pub(super) fn parse(mut args: Args) -> Result<Action, String> {
    let mut query = None;
    let mut config = FetchConfig {
        query: String::new(),
        ollama_database: false,
        lm_studio_database: false,
        list: false,
        file: None,
        output: PathBuf::new(),
    };
    let mut dry_run = false;
    while let Some(argument) = args.next() {
        let Some(flag) = Flag::parse(&argument) else {
            if query
                .replace(
                    argument
                        .into_string()
                        .map_err(|_| "model query must be UTF-8")?,
                )
                .is_some()
            {
                return Err(
                    "fetch accepts one model query; quote queries containing spaces".into(),
                );
            }
            continue;
        };
        match flag.name.as_str() {
            "--help" | "--help-all" | "-h" => {
                flag.switch()?;
                return Ok(Action::Help(help::FETCH));
            }
            "--ollama-database" => {
                flag.switch()?;
                config.ollama_database = true;
            }
            "--lm-studio-database" => {
                flag.switch()?;
                config.lm_studio_database = true;
            }
            "--list" => {
                flag.switch()?;
                config.list = true;
            }
            "--file" => config.file = Some(args.text(&flag)?),
            "--output" | "--out" | "-o" => {
                config.output = PathBuf::from(args.value(&flag)?);
                if config.output.as_os_str().is_empty() {
                    return Err("--out requires a nonempty directory; use --out . for the current directory".into());
                }
            }
            "--dry-run" => {
                flag.switch()?;
                dry_run = true;
            }
            _ => {
                return Err(format!(
                    "unknown fetch option '{}'; run `lma h f`",
                    flag.name
                ));
            }
        }
    }
    config.query = query
        .ok_or("fetch needs a model name or owner/repository; try `lma f llama3.2 -l`")?
        .trim()
        .to_owned();
    if config.query.is_empty()
        || config.query.len() > 512
        || config.query.chars().any(char::is_control)
    {
        return Err("model query must be 1..512 bytes without control characters".into());
    }
    if !config.ollama_database && !config.lm_studio_database {
        config.ollama_database = true;
        config.lm_studio_database = true;
    }
    if config.file.is_some() && !config.lm_studio_database {
        return Err("--file selects a Hugging Face GGUF file; use --lm-studio-database".into());
    }
    if config.output.as_os_str().is_empty() {
        config.output = crate::models::discovery::default_models_dir()?;
    }
    Ok(Action::Run(Box::new(Plan {
        job: Job::Fetch(config),
        dry_run,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::tests::plan;
    use std::ffi::OsString;

    #[test]
    fn combines_sources_by_default_and_honors_filters() {
        let Job::Fetch(config) = plan(&["f", "llama3.2", "-l"]).job else {
            panic!()
        };
        assert!(config.ollama_database && config.lm_studio_database && config.list);
        let Job::Fetch(config) = plan(&["nss", "fetch", "model", "--ollama-database"]).job else {
            panic!()
        };
        assert!(config.ollama_database && !config.lm_studio_database);
        let Job::Fetch(config) = plan(&[
            "nss",
            "fetch",
            "owner/repo",
            "--lm-studio-database",
            "--file",
            "model.gguf",
        ])
        .job
        else {
            panic!()
        };
        assert!(!config.ollama_database && config.lm_studio_database);
        assert_eq!(config.file.as_deref(), Some("model.gguf"));
    }

    #[test]
    fn canonical_source_flags_and_nss_output_aliases_are_preserved() {
        for argv in [
            vec![
                "f",
                "owner/repo",
                "-le",
                "-f=model.gguf",
                "-o=downloads",
                "-l",
                "-d",
            ],
            vec![
                "nss",
                "fetch",
                "owner/repo",
                "--lm-studio-database",
                "--file=model.gguf",
                "--out=downloads",
                "--list",
                "--dry-run",
            ],
            vec![
                "nss",
                "fetch",
                "owner/repo",
                "--lm-studio-database",
                "--file=model.gguf",
                "--output=downloads",
                "--list",
                "--dry-run",
            ],
            vec![
                "nss",
                "fetch",
                "owner/repo",
                "--lm-studio-database",
                "--file=model.gguf",
                "-o=downloads",
                "--list",
                "--dry-run",
            ],
        ] {
            let parsed = plan(&argv);
            assert!(parsed.dry_run);
            let Job::Fetch(config) = parsed.job else {
                panic!()
            };
            assert_eq!(config.query, "owner/repo");
            assert!(!config.ollama_database && config.lm_studio_database && config.list);
            assert_eq!(config.file.as_deref(), Some("model.gguf"));
            assert_eq!(config.output, PathBuf::from("downloads"));
        }
        let Job::Fetch(config) = plan(&["f", "model", "-oe", "-d"]).job else {
            panic!()
        };
        assert!(config.ollama_database && !config.lm_studio_database);
    }

    #[test]
    fn rejects_missing_or_conflicting_inputs() {
        for argv in [
            vec!["f"],
            vec!["f", "a", "b"],
            vec!["nss", "fetch", "m", "--ollama-database", "--file", "m.gguf"],
            vec!["nss", "fetch", "m", "--list=yes"],
        ] {
            assert!(crate::cli::parse(argv.into_iter().map(OsString::from)).is_err());
        }
    }
}
