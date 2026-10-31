//! Blocking deletion adapter. Only a fresh catalog ID crosses the IPC boundary;
//! roots are resolved by the host while settings/import publication is excluded.
use crate::{discovery::DeletionLocations, model_import, preferences, state::LoadOptions};
use lma::models::{self, DiscoveredModel, DiscoveryOptions};
use serde_json::Value;
use std::{fs, path::Path};

/// Call inside `AppState::delete_model`. The factory must resolve host settings,
/// not request-provided paths, and runs under the settings/import update lock.
pub(crate) fn run(
    config_dir: &Path,
    id: &str,
    locations: impl FnOnce() -> Result<DeletionLocations, String>,
) -> Result<Value, String> {
    if id.is_empty() || id.len() > 4096 || id.chars().any(char::is_control) {
        return Err("Invalid installed model ID.".into());
    }
    let _settings = model_import::SETTINGS_UPDATE
        .lock()
        .map_err(|_| "Settings update lock failed.")?;
    let locations = locations()?;
    // Discovery reports registry errors as warnings. Destructive resolution must
    // not silently ignore a registry that could contain another alias/reference.
    for root in &locations.local_dirs {
        models::registry::read(root)?;
    }
    let catalog = locations.catalog();
    let selected =
        catalog.models.iter().find(|model| model.id == id).ok_or(
            "That exact installed model is no longer available. Rescan models and try again.",
        )?;
    let alias = id
        .strip_prefix("lma:")
        .is_some_and(|name| models::registry::name(name).is_ok());
    // Desktop source labels describe the weights' location, not alias ownership.
    // A registered lma alias in an LM Studio folder is still deleted as an alias.
    let source = if alias { "lma" } else { &selected.source };
    let ollama_dir = if selected.ollama {
        selected.ollama_dir.clone()
    } else {
        // Protect custom Ollama blobs even when a configured local root overlaps
        // that store. Alias deletion may retain them; raw-file deletion must fail.
        locations
            .ollama_dirs
            .iter()
            .flatten()
            .filter_map(|root| fs::canonicalize(root).ok())
            .find(|root| Path::new(&selected.model).starts_with(root))
            .or_else(|| {
                catalog
                    .models
                    .iter()
                    .filter(|model| model.ollama)
                    .find_map(|model| {
                        model
                            .ollama_dir
                            .as_ref()
                            .filter(|root| Path::new(&selected.model).starts_with(root))
                            .cloned()
                    })
            })
    };
    let mut result = models::delete::run_in_locations(
        id,
        Some(source),
        DiscoveryOptions {
            local_dirs: locations.local_dirs,
            ollama_dir,
        },
        locations.lm_dirs,
    )?;
    if let Err(error) = cleanup_preference(config_dir, selected, &catalog.models, &result) {
        // Deletion already committed. Never report it as failed or invite a retry
        // merely because the saved choice could not be read/removed.
        if let Some(warnings) = result.get_mut("warnings").and_then(Value::as_array_mut) {
            warnings.push(Value::String(format!(
                "Model deleted, but the saved model selection could not be cleaned up: {error}"
            )));
        }
    }
    Ok(result)
}

fn cleanup_preference(
    config_dir: &Path,
    selected: &DiscoveredModel,
    catalog: &[DiscoveredModel],
    result: &Value,
) -> Result<(), String> {
    let Some(saved) = preferences::load(config_dir)? else {
        return Ok(());
    };
    let removed = |path: &Path| {
        result["deletedFiles"].as_array().is_some_and(|files| {
            files
                .iter()
                .filter_map(Value::as_str)
                .any(|file| Path::new(file) == path)
        })
    };
    let saved_path = if saved.model.starts_with("lma:") {
        catalog
            .iter()
            .find(|model| model.id == saved.model)
            .map(|model| Path::new(&model.model))
    } else if !saved.ollama {
        Some(Path::new(&saved.model))
    } else {
        None
    };
    let stale = (!saved.ollama && saved.model == selected.id && result["aliasDeleted"] == true)
        || (saved.ollama && selected.ollama && same_ollama_choice(&saved, selected))
        || saved_path.is_some_and(removed)
        || saved.projector.as_deref().is_some_and(removed);
    if stale {
        fs::remove_file(config_dir.join("model-preferences.json"))
            .map_err(|e| format!("Cannot remove saved model selection: {e}"))?;
    }
    Ok(())
}

fn same_ollama_choice(saved: &LoadOptions, selected: &DiscoveredModel) -> bool {
    let same_name = models::registry::name(&saved.model)
        .ok()
        .zip(models::registry::name(&selected.model).ok())
        .is_some_and(|(saved, selected)| saved == selected);
    let same_store = saved
        .ollama_dir
        .as_ref()
        .zip(selected.ollama_dir.as_ref())
        .is_some_and(|(saved, selected)| {
            fs::canonicalize(saved)
                .ok()
                .zip(fs::canonicalize(selected).ok())
                .is_some_and(|(saved, selected)| saved == selected)
        });
    same_name && same_store
}
