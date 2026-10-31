mod chat;
pub mod clients;
mod compat;
mod create;
mod delete;
mod embeddings;
mod expert_history;
mod hosting;
mod imagegen;
mod inference;
pub(crate) mod launch;
mod list;
mod ollama;
pub mod quantize;
pub mod server;
// Candidate-history prototype only; verified batched decoding is not integrated.
#[cfg(test)]
mod speculation;
mod stop;
mod tools;
mod web;

use crate::config::{Job, Plan, ServeConfig};

pub fn run(mut plan: Plan) -> Result<(), String> {
    if let Job::Serve(config) = &mut plan.job {
        let explicit_ollama = config
            .model
            .to_str()
            .is_some_and(|path| path.starts_with("ollama:"))
            || config.ollama_dir.is_some();
        let explicit_profile = config
            .model
            .to_str()
            .is_some_and(|model| model.starts_with("lma:"));
        if !plan.dry_run || explicit_ollama || explicit_profile {
            resolve_model(config)?;
        }
    }
    if matches!(&plan.job, Job::Create(_)) {
        let Job::Create(config) = plan.job else {
            unreachable!()
        };
        return create::run(config, plan.dry_run);
    }
    if let Job::Import(config) = &plan.job
        && plan.dry_run
    {
        let preview = crate::import::preview(config)?;
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({"command":"import", "dry_run":true, "options":config, "summary":preview})).map_err(|e| e.to_string())?);
        return Ok(());
    }
    if plan.dry_run {
        println!(
            "{}",
            serde_json::to_string_pretty(&plan)
                .map_err(|error| format!("cannot preview configuration: {error}"))?
        );
        return Ok(());
    }
    match plan.job {
        Job::Serve(config) => tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|error| format!("cannot start async runtime: {error}"))?
            .block_on(server::run(config)),
        Job::Quantize(config) => {
            let output = config.output.clone();
            quantize::run(config)?;
            eprintln!("Wrote {}", output.display());
            Ok(())
        }
        Job::Stop(config) => stop::run(config),
        Job::Fetch(config) => crate::fetch::run(config),
        Job::Image(config) => imagegen::run(config),
        Job::Launch(config) => crate::tui::run(config),
        Job::List(config) => list::run(config),
        Job::Import(config) => crate::import::run(config),
        Job::Create(config) => create::run(config, false),
        Job::Delete(config) => delete::run(config),
        Job::Transcribe(config) => {
            let transcript = crate::speech::transcribe(config.options())?;
            if config.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&transcript).map_err(|e| e.to_string())?
                );
            } else {
                println!("{}", transcript.text);
            }
            Ok(())
        }
    }
}

pub(crate) fn resolve_model(config: &mut ServeConfig) -> Result<(), String> {
    resolve_model_metadata(config)?;
    create::apply_load_defaults(config)
}

/// Resolve installed identity and definition without applying load defaults or
/// validating the effective load configuration. Hosts may apply overrides first.
pub(crate) fn resolve_model_metadata(config: &mut ServeConfig) -> Result<(), String> {
    let input_is_file = config.model.is_file();
    let resolved = crate::models::resolve(&config.model, config.ollama_dir.as_deref())?;
    apply_model_metadata(
        config,
        resolved,
        input_is_file,
        &crate::models::discovery::default_local_dirs(),
    )
}

fn apply_model_metadata(
    config: &mut ServeConfig,
    resolved: crate::models::ResolvedModel,
    input_is_file: bool,
    registry_roots: &[std::path::PathBuf],
) -> Result<(), String> {
    if !input_is_file {
        config.definition = resolved.registry_definition(registry_roots)?;
    }
    if let Some(name) = &resolved.name
        && resolved
            .model
            .parent()
            .is_some_and(|parent| parent.ends_with("blobs"))
    {
        eprintln!(
            "Using installed Ollama model {name}; Ollama template/system/parameter layers are not applied"
        );
    }
    config.model = resolved.model;
    config.model_id = resolved.name;
    if config.projector.is_none() {
        config.projector = resolved.projector;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ResolvedModel, modelfile, registry};
    use std::{fs, path::Path};

    fn resolved(model: &Path, projector: Option<&Path>, registered: bool) -> ResolvedModel {
        ResolvedModel {
            model: model.into(),
            projector: projector.map(Path::to_path_buf),
            name: Some("custom:latest".into()),
            registry_name: registered.then(|| "custom:latest".into()),
        }
    }

    fn publish(
        root: &Path,
        model: &Path,
        projector: Option<&Path>,
        system: &str,
    ) -> modelfile::Definition {
        let definition = modelfile::parse(&format!(
            "FROM weights\nSYSTEM {system}\nTEMPLATE chatml\nPARAMETER num_ctx 512\nPARAMETER num_batch 32"
        ))
        .unwrap();
        registry::put_definition(
            root,
            "custom",
            registry::Entry {
                path: model.into(),
                projector: projector.map(Path::to_path_buf),
            },
            definition.clone(),
        )
        .unwrap();
        definition
    }

    #[test]
    fn registry_definition_comes_from_matching_weights_and_projector_not_default_root() {
        let dir = tempfile::tempdir().unwrap();
        let weights = dir.path().join("weights.gguf");
        let other = dir.path().join("other.gguf");
        let projector = dir.path().join("projector.gguf");
        let other_projector = dir.path().join("other-projector.gguf");
        for path in [&weights, &other, &projector, &other_projector] {
            fs::write(path, b"GGUF").unwrap();
        }
        let roots = vec![
            dir.path().join("default"),
            dir.path().join("wrong-projector"),
            dir.path().join("selected"),
        ];
        publish(&roots[0], &other, Some(&projector), "wrong-weights");
        publish(
            &roots[1],
            &weights,
            Some(&other_projector),
            "wrong-projector",
        );
        let definition = publish(&roots[2], &weights, Some(&projector), "selected");
        let explicit_projector = dir.path().join("explicit.gguf");
        let mut config = ServeConfig {
            model: "custom".into(),
            projector: Some(explicit_projector.clone()),
            ..ServeConfig::default()
        };
        apply_model_metadata(
            &mut config,
            resolved(&weights, Some(&projector), true),
            false,
            &roots,
        )
        .unwrap();
        assert_eq!(config.model, weights);
        assert_eq!(config.model_id.as_deref(), Some("custom:latest"));
        assert_eq!(config.projector, Some(explicit_projector));
        assert_eq!(config.definition.as_ref(), Some(&definition));
        assert_eq!(config.context, 4096);
        assert_eq!(config.batch, 512);
        assert_eq!(config.chat_template, "auto");
        create::apply_load_defaults(&mut config).unwrap();
        assert_eq!(config.context, 512);
        assert_eq!(config.batch, 32);
        assert_eq!(config.micro_batch, 32);
        assert_eq!(config.chat_template, "chatml");
    }

    #[test]
    fn same_name_without_registry_provenance_never_loads_a_definition() {
        let dir = tempfile::tempdir().unwrap();
        let weights = dir.path().join("weights.gguf");
        fs::write(&weights, b"GGUF").unwrap();
        let root = dir.path().join("registry");
        let definition = publish(&root, &weights, None, "must-not-leak");
        for selector in ["custom", "ollama:custom"] {
            let mut config = ServeConfig {
                model: selector.into(),
                definition: Some(definition.clone()),
                ..ServeConfig::default()
            };
            apply_model_metadata(
                &mut config,
                resolved(&weights, None, false),
                false,
                std::slice::from_ref(&root),
            )
            .unwrap();
            assert_eq!(config.definition, None, "{selector}");
            assert_eq!(config.context, 4096);
        }
    }

    #[test]
    fn changed_registry_files_fail_instead_of_attaching_a_same_name_definition() {
        let dir = tempfile::tempdir().unwrap();
        let selected = dir.path().join("selected.gguf");
        let other = dir.path().join("other.gguf");
        fs::write(&selected, b"GGUF").unwrap();
        fs::write(&other, b"GGUF").unwrap();
        let root = dir.path().join("registry");
        publish(&root, &other, None, "wrong");
        let mut config = ServeConfig {
            model: "custom".into(),
            ..ServeConfig::default()
        };
        let error =
            apply_model_metadata(&mut config, resolved(&selected, None, true), false, &[root])
                .unwrap_err();
        assert!(error.contains("no longer matches"), "{error}");
        assert_eq!(config.definition, None);
        assert_eq!(config.model, Path::new("custom"));
    }

    #[test]
    fn legacy_alias_does_not_inherit_a_later_registry_definition() {
        let dir = tempfile::tempdir().unwrap();
        let weights = dir.path().join("weights.gguf");
        fs::write(&weights, b"GGUF").unwrap();
        let roots = vec![dir.path().join("first"), dir.path().join("second")];
        registry::put(
            &roots[0],
            "custom",
            registry::Entry {
                path: weights.clone(),
                projector: None,
            },
        )
        .unwrap();
        publish(&roots[1], &weights, None, "later");
        let mut config = ServeConfig::default();
        apply_model_metadata(&mut config, resolved(&weights, None, true), false, &roots).unwrap();
        assert_eq!(config.definition, None);
    }

    #[test]
    fn metadata_only_populates_discovered_projector_without_load_validation() {
        let dir = tempfile::tempdir().unwrap();
        let weights = dir.path().join("weights.gguf");
        let projector = dir.path().join("projector.gguf");
        fs::write(&weights, b"GGUF").unwrap();
        fs::write(&projector, b"GGUF").unwrap();
        let root = dir.path().join("registry");
        publish(&root, &weights, Some(&projector), "selected");
        let mut config = ServeConfig {
            model: "custom".into(),
            context: 1,
            explicit_load: vec!["num_ctx".into()],
            ..ServeConfig::default()
        };
        apply_model_metadata(
            &mut config,
            resolved(&weights, Some(&projector), true),
            false,
            &[root],
        )
        .unwrap();
        assert_eq!(config.projector, Some(projector));
        assert_eq!(config.context, 1);
        assert!(create::apply_load_defaults(&mut config).is_err());
    }

    #[test]
    fn local_metadata_preserves_definition_without_applying_or_validating_it() {
        let dir = tempfile::tempdir().unwrap();
        let weights = dir.path().join("weights.gguf");
        fs::write(&weights, b"GGUF").unwrap();
        let definition = modelfile::Definition::default(); // Invalid until FROM is supplied.
        let mut config = ServeConfig {
            model: weights.clone(),
            definition: Some(definition.clone()),
            ..ServeConfig::default()
        };
        resolve_model_metadata(&mut config).unwrap();
        assert_eq!(config.model, weights);
        assert_eq!(config.definition.as_ref(), Some(&definition));
        assert!(resolve_model(&mut config).is_err());
    }

    #[test]
    fn full_resolution_still_applies_local_definition_defaults_and_explicit_overrides() {
        let dir = tempfile::tempdir().unwrap();
        let weights = dir.path().join("weights.gguf");
        fs::write(&weights, b"GGUF").unwrap();
        let mut config = ServeConfig {
            model: weights,
            definition: Some(
                modelfile::parse(
                    "FROM weights\nTEMPLATE chatml\nPARAMETER num_ctx 512\nPARAMETER num_batch 32",
                )
                .unwrap(),
            ),
            context: 1024,
            explicit_load: vec!["num_ctx".into()],
            ..ServeConfig::default()
        };
        resolve_model_metadata(&mut config).unwrap();
        assert_eq!(config.chat_template, "auto");
        assert_eq!(config.batch, 512);
        resolve_model(&mut config).unwrap();
        assert_eq!(config.chat_template, "chatml");
        assert_eq!(config.context, 1024);
        assert_eq!(config.batch, 32);
        assert_eq!(config.micro_batch, 32);
    }
}
