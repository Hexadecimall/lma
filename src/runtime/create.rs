//! Explicit Modelfile registration. Only metadata is published; weights stay put.
use crate::{
    config::{CreateConfig, ServeConfig},
    models::{
        self,
        modelfile::{self, Definition},
        registry::{self, Entry},
    },
};
use std::path::Path;

pub(super) fn inherit_definition(
    mut child: Definition,
    base: Option<Definition>,
) -> Result<Definition, String> {
    child.validate()?;
    if let Some(mut base) = base {
        base.validate()?;
        child.system = child.system.or(base.system);
        child.template = child.template.or(base.template);
        child.license = child.license.or(base.license);
        base.messages.append(&mut child.messages);
        child.messages = base.messages;
        let parent = base.parameters;
        let values = &mut child.parameters;
        values.temperature = values.temperature.or(parent.temperature);
        values.top_p = values.top_p.or(parent.top_p);
        values.seed = values.seed.or(parent.seed);
        values.num_predict = values.num_predict.or(parent.num_predict);
        values.num_ctx = values.num_ctx.or(parent.num_ctx);
        values.num_gpu = values.num_gpu.or(parent.num_gpu);
        values.num_thread = values.num_thread.or(parent.num_thread);
        values.num_batch = values.num_batch.or(parent.num_batch);
        values.u_batch = values.u_batch.or(parent.u_batch);
    }
    child.validate()?;
    Ok(child)
}

fn prepare_at(config: &CreateConfig, root: &Path) -> Result<(Definition, Entry), String> {
    registry::name(&config.model)?;
    let mut inherited = None;
    let (definition, entry) = modelfile::prepare(&config.file, |from| {
        let input = Path::new(from);
        if !input.is_file() && !from.starts_with("ollama:") {
            let label = from.strip_prefix("lma:").unwrap_or(from);
            if let Ok(label) = registry::name(label)
                && let Some(entry) = registry::read(root)?.get(&label).cloned()
            {
                inherited = registry::definition(root, &label)?;
                return Ok(entry);
            }
        }
        let resolved = models::resolve(input, config.ollama_dir.as_deref())?;
        Ok(Entry {
            path: resolved.model,
            projector: resolved.projector,
        })
    })?;
    Ok((
        inherit_definition(definition, inherited)?,
        registry::canonical_entry(entry)?,
    ))
}

pub(super) fn run(config: CreateConfig, dry_run: bool) -> Result<(), String> {
    let root = models::discovery::default_models_dir()?;
    let (definition, entry) = prepare_at(&config, &root)?;
    if dry_run {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "command":"create", "dry_run":true, "model":registry::name(&config.model)?,
                "file":config.file, "weights":entry.path, "projector":entry.projector,
                "definition":definition,
            }))
            .map_err(|error| error.to_string())?
        );
    } else {
        let label = registry::put_definition(&root, &config.model, entry, definition)?;
        println!("{}", serde_json::json!({"status":"success", "model":label}));
    }
    Ok(())
}

/// Apply named-model load defaults, keeping explicit CLI/application values.
pub(super) fn apply_load_defaults(config: &mut ServeConfig) -> Result<(), String> {
    let Some(definition) = &config.definition else {
        return config.apply_expert_cache_policy();
    };
    definition.validate()?;
    let implicit = |name: &str| !config.explicit_load.iter().any(|key| key == name);
    let parameters = &definition.parameters;
    if implicit("template")
        && let Some(template) = &definition.template
    {
        config.chat_template = template.clone();
    }
    if implicit("num_ctx")
        && let Some(context) = parameters.num_ctx
    {
        config.context = context;
    }
    if implicit("num_gpu")
        && let Some(layers) = parameters.num_gpu
    {
        config.gpu_layers = layers;
    }
    if implicit("num_thread")
        && let Some(threads) = parameters.num_thread
    {
        config.threads = threads;
    }
    if implicit("num_batch")
        && let Some(batch) = parameters.num_batch
    {
        config.batch = batch;
    }
    if implicit("u_batch") {
        config.micro_batch = parameters
            .u_batch
            .unwrap_or(config.micro_batch.min(config.batch));
    }
    if config.context < 2 || config.context > i32::MAX as u32 {
        return Err("effective model context must be between 2 and i32::MAX".into());
    }
    if parameters
        .num_predict
        .is_some_and(|tokens| tokens > config.context)
    {
        return Err("Modelfile num_predict exceeds the effective context; reduce num_predict or increase -c".into());
    }
    config.apply_expert_cache_policy()?;
    if config.micro_batch > config.batch {
        return Err("effective Modelfile/explicit u_batch must not exceed num_batch; set -mh or adjust the definition".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn prepare_inherits_metadata_and_references_weights_without_publishing() {
        let dir = tempfile::tempdir().unwrap();
        let weights = dir.path().join("weights.gguf");
        fs::write(&weights, b"GGUFfixture").unwrap();
        let root = dir.path().join("registry");
        let base = modelfile::parse("FROM weights\nSYSTEM base\nPARAMETER temperature 0.5\nPARAMETER num_ctx 512\nMESSAGE user example").unwrap();
        registry::put_definition(
            &root,
            "base",
            Entry {
                path: weights.clone(),
                projector: None,
            },
            base,
        )
        .unwrap();
        let file = dir.path().join("Modelfile");
        fs::write(
            &file,
            "FROM base\nPARAMETER temperature 0\nMESSAGE assistant reply",
        )
        .unwrap();
        let config = CreateConfig {
            model: "child".into(),
            file,
            ollama_dir: None,
        };
        let before = fs::read(root.join(".registry-v1.json")).unwrap();
        let (definition, entry) = prepare_at(&config, &root).unwrap();
        assert_eq!(definition.system.as_deref(), Some("base"));
        assert_eq!(definition.parameters.temperature, Some(0.0));
        assert_eq!(definition.parameters.num_ctx, Some(512));
        assert_eq!(definition.messages.len(), 2);
        assert_eq!(entry.path, fs::canonicalize(&weights).unwrap());
        assert_eq!(fs::read(root.join(".registry-v1.json")).unwrap(), before);
        assert!(!registry::read(&root).unwrap().contains_key("child:latest"));
        registry::put_definition(&root, &config.model, entry, definition).unwrap();
        assert_eq!(fs::read(&weights).unwrap(), b"GGUFfixture");
        assert_eq!(registry::read(&root).unwrap().len(), 2);
        assert_eq!(fs::read_dir(&root).unwrap().count(), 2);
    }

    #[test]
    fn load_defaults_preserve_explicit_values_even_when_equal_to_original_defaults() {
        let definition = modelfile::parse("FROM x\nTEMPLATE chatml\nPARAMETER num_ctx 128\nPARAMETER num_gpu 0\nPARAMETER num_thread 2\nPARAMETER num_batch 32\nPARAMETER u_batch 16").unwrap();
        let mut implicit = ServeConfig {
            definition: Some(definition.clone()),
            ..Default::default()
        };
        apply_load_defaults(&mut implicit).unwrap();
        assert_eq!(
            (
                implicit.context,
                implicit.gpu_layers,
                implicit.threads,
                implicit.batch,
                implicit.micro_batch
            ),
            (128, 0, 2, 32, 16)
        );
        assert_eq!(implicit.chat_template, "chatml");
        let mut explicit = ServeConfig {
            definition: Some(definition),
            explicit_load: [
                "template",
                "num_ctx",
                "num_gpu",
                "num_thread",
                "num_batch",
                "u_batch",
            ]
            .map(String::from)
            .into(),
            ..Default::default()
        };
        let defaults = ServeConfig::default();
        apply_load_defaults(&mut explicit).unwrap();
        assert_eq!(
            (
                explicit.context,
                explicit.gpu_layers,
                explicit.threads,
                explicit.batch,
                explicit.micro_batch
            ),
            (
                defaults.context,
                defaults.gpu_layers,
                defaults.threads,
                defaults.batch,
                defaults.micro_batch
            )
        );
        assert_eq!(explicit.chat_template, "auto");
    }

    #[test]
    fn expert_cache_preserves_profile_batching_and_explicit_micro_batch_overrides() {
        let definition =
            modelfile::parse("FROM x\nPARAMETER num_batch 32\nPARAMETER u_batch 16").unwrap();
        let mut config = ServeConfig {
            expert_cache_mib: 1024,
            definition: Some(definition),
            ..Default::default()
        };
        apply_load_defaults(&mut config).unwrap();
        assert_eq!(config.micro_batch, 16);
        assert_eq!(config.batch, 32);
        config.micro_batch = 2;
        config.explicit_load.push("u_batch".into());
        apply_load_defaults(&mut config).unwrap();
        assert_eq!(config.micro_batch, 2);
    }

    #[test]
    fn invalid_merged_batch_and_output_budgets_are_rejected() {
        let base = modelfile::parse("FROM x\nPARAMETER u_batch 64").unwrap();
        let child = modelfile::parse("FROM base\nPARAMETER num_batch 32").unwrap();
        assert!(inherit_definition(child, Some(base)).is_err());
        let mut config = ServeConfig {
            context: 64,
            definition: Some(modelfile::parse("FROM x\nPARAMETER num_predict 128").unwrap()),
            ..Default::default()
        };
        assert!(apply_load_defaults(&mut config).is_err());
    }
}
