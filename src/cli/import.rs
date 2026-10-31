use super::{
    Action,
    args::{Args, Flag},
    help,
};
use crate::config::{ImportConfig, Job, Plan};
use std::path::PathBuf;

pub(super) fn parse(mut args: Args) -> Result<Action, String> {
    let mut model = None;
    let mut output = None;
    let mut ollama_dir = None;
    let mut dry_run = false;
    let mut hf_dir = false;
    while let Some(arg) = args.next() {
        if let Some(flag) = Flag::parse(&arg) {
            match flag.name.as_str() {
                "--help" | "--help-all" | "-h" => {
                    flag.switch()?;
                    return Ok(Action::Help(help::IMPORT));
                }
                "--ollama" | "-o" | "--model" | "-m" | "--hf-dir" => {
                    hf_dir = flag.name == "--hf-dir";
                    if model.replace(args.text(&flag)?).is_some() {
                        return Err("import model specified twice".into());
                    }
                }
                "--ollama-dir" | "-od" => ollama_dir = Some(PathBuf::from(args.value(&flag)?)),
                "--out" | "--output" => output = Some(PathBuf::from(args.value(&flag)?)),
                "--dry-run" => {
                    flag.switch()?;
                    dry_run = true;
                }
                _ => {
                    return Err(format!(
                        "unknown import option '{}'; use `lma h it`",
                        flag.name
                    ));
                }
            }
        } else if model
            .replace(
                arg.into_string()
                    .map_err(|_| "import model reference must be UTF-8")?,
            )
            .is_some()
        {
            return Err("import accepts one Ollama model reference or local HF directory".into());
        }
    }
    let model = model.ok_or("import needs an Ollama tag or local HF directory: `lma it -hr ./model -ot models/model.gguf -d` or `lma nss import --hf-dir ./model --out models/model.gguf --dry-run`")?;
    if model.is_empty() || model.len() > 1024 || model.chars().any(char::is_control) {
        return Err("invalid import model reference".into());
    }
    if hf_dir && !std::path::Path::new(&model).is_dir() {
        return Err("--hf-dir requires an existing local model directory".into());
    }
    if std::path::Path::new(&model).is_dir() && ollama_dir.is_some() {
        return Err("a local HF directory cannot be combined with --ollama-dir".into());
    }
    let output =
        output.ok_or("import requires an output path: use -ot FILE.gguf and -d at root, or --out FILE.gguf and --dry-run in NSS")?;
    if !output
        .extension()
        .is_some_and(|v| v.eq_ignore_ascii_case("gguf"))
    {
        return Err("import output must end in .gguf".into());
    }
    Ok(Action::Run(Box::new(Plan {
        job: Job::Import(ImportConfig {
            model,
            ollama_dir,
            output,
        }),
        dry_run,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::tests::plan;
    #[test]
    fn local_hf_directory_uses_existing_model_field() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        for source in [
            vec![
                "nss",
                "import",
                "--hf-dir",
                path,
                "--out",
                "model.gguf",
                "--dry-run",
            ],
            vec!["nss", "import", path, "--out", "model.gguf", "--dry-run"],
            vec!["it", "-hr", path, "-ot", "model.gguf", "-d"],
            vec!["it", path, "-ot", "model.gguf", "-d"],
        ] {
            let p = plan(&source);
            let Job::Import(c) = p.job else { panic!() };
            assert_eq!(c.model, path);
            assert!(c.ollama_dir.is_none());
            assert!(p.dry_run);
        }
        for source in [
            vec![
                "nss",
                "import",
                "--hf-dir",
                "./missing-hf-import-fixture",
                "--out",
                "model.gguf",
            ],
            vec![
                "nss",
                "import",
                "--hf-dir",
                path,
                "--ollama-dir",
                "store",
                "--out",
                "model.gguf",
            ],
        ] {
            assert!(crate::cli::parse(source.into_iter().map(Into::into)).is_err());
        }
    }
    #[test]
    fn canonical_import_flags_and_nss_output_aliases_are_preserved() {
        for argv in [
            vec![
                "it",
                "-o",
                "local:latest",
                "-or=store",
                "-ot",
                "model.gguf",
                "-d",
            ],
            vec![
                "it",
                "-m",
                "local:latest",
                "-or=store",
                "-ot",
                "model.gguf",
                "-d",
            ],
            vec![
                "nss",
                "import",
                "--model",
                "local:latest",
                "--ollama-dir=store",
                "--output",
                "model.gguf",
                "--dry-run",
            ],
            vec![
                "nss",
                "import",
                "--ollama",
                "local:latest",
                "-od=store",
                "--out",
                "model.gguf",
                "--dry-run",
            ],
        ] {
            let parsed = plan(&argv);
            assert!(parsed.dry_run);
            let Job::Import(config) = parsed.job else {
                panic!()
            };
            assert_eq!(config.model, "local:latest");
            assert_eq!(config.ollama_dir, Some(PathBuf::from("store")));
            assert_eq!(config.output, PathBuf::from("model.gguf"));
        }
    }

    #[test]
    fn import_is_explicit_and_keeps_ollama_shortcuts_in_nss() {
        let p = plan(&[
            "nss",
            "import",
            "-o",
            "gemma4:12b",
            "-od=store",
            "--out",
            "models/gemma.gguf",
            "--dry-run",
        ]);
        let Job::Import(config) = p.job else { panic!() };
        assert!(p.dry_run);
        assert_eq!(config.model, "gemma4:12b");
        assert_eq!(config.ollama_dir, Some("store".into()));
        assert_eq!(config.output, PathBuf::from("models/gemma.gguf"));
        for argv in [
            vec!["it"],
            vec!["it", "gemma4"],
            vec!["nss", "import", "gemma4", "--out=model.bin"],
            vec!["nss", "import", "a", "b", "--out=a.gguf"],
        ] {
            assert!(crate::cli::parse(argv.into_iter().map(Into::into)).is_err());
        }
    }
}
