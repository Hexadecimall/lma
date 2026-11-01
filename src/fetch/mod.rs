//! Public, unauthenticated GGUF fetching. No daemon, subprocess, git, or ghx is
//! invoked. The CLI owns aliases/source-flag parsing and the default output path.
//!
//! Downloads are selected by exact source identity, verified against upstream
//! SHA256 and size metadata, and published without replacing existing files.
//! Output directories are assumed to be locally trusted: symlinks beneath the
//! chosen root are refused, but this is not a sandbox against a local process
//! concurrently replacing directories. GGUF magic is checked, not architecture
//! compatibility or the full GGUF metadata/tensor structure.

mod catalog;
mod download;
pub mod huggingface;
mod progress;
pub mod store;

use crate::config::FetchConfig;
use catalog::{Candidate, Catalog, Source};

/// Discover/list public GGUF sources or fetch one unambiguous exact match.
///
/// `--list` never downloads weights. A bare Ollama name means `library/name:latest`;
/// Hugging Face requires `owner/repository` to download, otherwise it only searches.
/// Both sources are consulted independently; an exact lookup failure blocks an
/// automatic download even when another source succeeds. No authentication is sent.
pub fn run(config: FetchConfig) -> Result<(), String> {
    let query = config.query.trim();
    if !(1..=1000).contains(&config.index) {
        return Err("index must be an integer between 1 and 1000 (number of results)".into());
    }
    if (!config.list && query.is_empty())
        || query.len() > 512
        || query.chars().any(char::is_control)
    {
        return Err(
            "fetch requires a nonempty model query of at most 512 bytes without control characters"
                .into(),
        );
    }
    if let Some(file) = &config.file {
        download::safe_relative(file)?;
    }
    let client = download::client()?;
    // The CLI already defaults to both; retain that behavior for direct callers.
    let both = !config.ollama_database && !config.lm_studio_database;
    if query.is_empty() {
        if !config.lm_studio_database && !both {
            return Err("popularity browsing uses Hugging Face download counts; Ollama-only listing needs a model query".into());
        }
        if config.file.is_some() {
            return Err("a Hugging Face filename needs an owner/repository query".into());
        }
        let models = catalog::popular(&client, config.index)?;
        println!("Popular public Hugging Face GGUF repositories — ranked by reported downloads.");
        println!(
            "Ollama has no comparable popularity ranking here. Catalog entries are not a runtime compatibility guarantee."
        );
        println!("RANK  DOWNLOADS  REPOSITORY");
        for (index, model) in models.iter().enumerate() {
            println!("{:>4}  {:>9}  {}", index + 1, model.downloads, model.id);
        }
        if models.is_empty() {
            println!("No popular GGUF repositories returned by Hugging Face.");
        }
        return Ok(());
    }
    let discover = if config.list {
        catalog::discover_listing
    } else {
        catalog::discover_with_limit
    };
    let catalog = discover(
        &client,
        query,
        config.ollama_database || both,
        config.lm_studio_database || both,
        config.index,
    );
    for warning in &catalog.warnings {
        eprintln!("Warning: {warning}");
    }
    if catalog.succeeded == 0 {
        return Err(format!(
            "all requested catalog sources failed: {}",
            catalog.warnings.join("; ")
        ));
    }
    print_choices(&catalog, config.index);
    if config.list {
        return Ok(());
    }
    let candidate = exact_candidate(&catalog)?;
    let selection = catalog::select(candidate, config.file.as_deref())?;
    // Resolve/validate every descriptor before any weights are downloaded.
    let directory = download::directory(&config.output, &selection.relative_directory)?;
    download::save_metadata(&directory, selection.metadata_filename, &selection.metadata)?;
    for note in &selection.notes {
        println!("{note}");
    }
    let mut usable = Vec::new();
    for artifact in &selection.artifacts {
        let path = download::download_cli(&client, &directory, artifact)?;
        if artifact.gguf {
            usable.push(path);
        }
    }
    if !selection.projector_only
        && candidate.source == Source::Ollama
        && let Some(model) = usable.first()
    {
        let label = crate::models::registry::name(&candidate.id)?;
        if !crate::models::registry::read(&config.output)?.contains_key(&label) {
            crate::models::registry::put(
                &config.output,
                &label,
                crate::models::registry::Entry {
                    path: model.clone(),
                    projector: usable.get(1).cloned(),
                },
            )?;
        }
    }
    for path in &usable {
        println!("GGUF path: {}", path.display());
    }
    if selection.projector_only {
        println!(
            "Projector downloaded; supply this path with -pr when serving its compatible model (`lma s MODEL -pr FILE`). No standalone server command is available for a projector."
        );
    } else if let Some(model) = usable.first() {
        let mut suggestion = format!("lma s -p {}", shell_quote(&model.to_string_lossy()));
        if candidate.source == Source::Ollama
            && let Some(projector) = usable.get(1)
        {
            suggestion.push_str(&format!(
                " -pr {}",
                shell_quote(&projector.to_string_lossy())
            ));
        }
        println!(
            "Try (not executed; requires a runtime-supported GGUF architecture):\n  {suggestion}"
        );
    }
    Ok(())
}

/// Ollama HTTP pull adapter: same verified in-process downloader, fixed lma root.
pub(crate) fn pull_ollama(
    query: &str,
    root: &std::path::Path,
) -> Result<crate::models::registry::Entry, String> {
    let client = download::client()?;
    let catalog = catalog::discover(&client, query, true, false);
    let candidate = exact_candidate(&catalog)?;
    let selection = catalog::select(candidate, None)?;
    if selection.projector_only {
        return Err("cannot pull a projector as a standalone model".into());
    }
    let directory = download::directory(root, &selection.relative_directory)?;
    download::save_metadata(&directory, selection.metadata_filename, &selection.metadata)?;
    let mut paths = Vec::new();
    for artifact in &selection.artifacts {
        paths.push(download::download(&client, &directory, artifact)?);
    }
    let path = paths
        .first()
        .cloned()
        .ok_or("model download contained no weights")?;
    Ok(crate::models::registry::Entry {
        path,
        projector: paths.get(1).cloned(),
    })
}

fn print_choices(catalog: &Catalog, limit: u32) {
    if catalog.candidates.is_empty() {
        println!(
            "No catalog matches. Ollama supports exact name[:tag] or namespace/name[:tag] lookups; the LM Studio source searches public Hugging Face GGUF repositories."
        );
    }
    for candidate in catalog.candidates.iter().take(limit as usize) {
        println!(
            "[{}] {}{} ({})",
            candidate.source.label(),
            candidate.id,
            if candidate.exact {
                " [exact]"
            } else {
                " [search result]"
            },
            candidate.source.flag()
        );
    }
}

fn exact_candidate(catalog: &Catalog) -> Result<&Candidate, String> {
    if catalog.failed_exact {
        return Err("an exact source lookup failed (see warnings); refusing to choose another source implicitly. Retry, or specify the full repository/reference with only -oe or -le".into());
    }
    let exact: Vec<_> = catalog
        .candidates
        .iter()
        .filter(|candidate| candidate.exact)
        .collect();
    match exact.as_slice() {
        [candidate] => Ok(candidate),
        [] => Err("no exact downloadable match; choose a full Hugging Face owner/repository or Ollama name[:tag] from the choices, and specify -le or -oe. Search results are never downloaded automatically".into()),
        _ => Err("multiple exact source matches; choose a full repository/reference and restrict the source with -le or -oe".into()),
    }
}

fn shell_quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(source: Source, exact: bool) -> Candidate {
        Candidate {
            source,
            id: "owner/repo".into(),
            exact,
            metadata: None,
        }
    }

    #[test]
    fn selection_requires_one_exact_and_no_failed_exact_lookup() {
        let mut catalog = Catalog {
            succeeded: 1,
            ..Catalog::default()
        };
        assert!(exact_candidate(&catalog).is_err());
        catalog.candidates.push(candidate(Source::LmStudio, false));
        assert!(exact_candidate(&catalog).is_err());
        catalog.candidates.push(candidate(Source::Ollama, true));
        assert_eq!(exact_candidate(&catalog).unwrap().source, Source::Ollama);
        catalog.failed_exact = true;
        assert!(exact_candidate(&catalog).is_err());
        catalog.failed_exact = false;
        catalog.candidates.push(candidate(Source::LmStudio, true));
        assert!(exact_candidate(&catalog).is_err());
    }

    #[test]
    fn server_suggestion_quotes_paths_without_execution() {
        assert_eq!(
            shell_quote("models/my model.gguf"),
            "'models/my model.gguf'"
        );
        assert_eq!(shell_quote("it's.gguf"), "'it'\\''s.gguf'");
    }
}
