//! Opt-in, local-only regression. Run this test alone (`--test-threads=1`):
//! set LMA_TEST_TEXT_GGUF to a small installed text GGUF, or
//! LMA_TEST_OLLAMA_MODEL to an installed Ollama selector (optional `ollama:` prefix).
//! OLLAMA_MODELS selects the store as usual. No download, daemon, or subprocess.
//! Expert-cache coverage requires a supported qwen2moe/qwen3moe/mellum layout.
//! LMA_TEST_EXPERT_CACHE_MIB defaults to 1024 (must fit at least active top-k);
//! LMA_TEST_EXPERT_CACHE_GPU_LAYERS defaults to 0 (explicit GPU-slot opt-in).
//! LMA_TEST_EXPERT_CACHE_MICRO_BATCH defaults to 1; use 2..=32 to check prefill.
//! LMA_TEST_EXPERT_CACHE_LONG_PROMPT adds a prompt spanning several 16-token tiles.
//! LMA_TEST_TEXT_GPU_LAYERS selects placement for the until-context regression (default CPU).

use super::*;
use std::{
    cell::Cell,
    path::PathBuf,
    time::{Duration, Instant},
};

fn text_request(text: &str) -> GenerationRequest {
    GenerationRequest {
        prompt: Prompt::Text(text.into()),
        options: Options {
            until_context: false,
            max_tokens: 4,
            temperature: 0.0,
            top_p: 1.0,
            seed: Some(42),
        },
    }
}

fn generate_text(
    engine: &mut LlamaEngine,
    text: &str,
    label: &str,
    cancelled: &dyn Fn() -> bool,
) -> Generation {
    let started = Instant::now();
    let result = engine.generate(text_request(text), cancelled);
    eprintln!("{label}: elapsed={:?} (diagnostic only)", started.elapsed());
    result.unwrap_or_else(|error| panic!("{label}: {error:?}"))
}

fn assert_same_generation(expected: &Generation, actual: &Generation, label: &str) {
    assert_generation_with_limit(expected, actual, 4, label);
}

fn assert_generation_with_limit(
    expected: &Generation,
    actual: &Generation,
    max_tokens: usize,
    label: &str,
) {
    // Compare raw text, including whitespace; neither trimming nor tolerances
    // may hide a cache/reset-induced change in deterministic greedy selection.
    assert_eq!(expected.text, actual.text, "{label}: raw text");
    assert_eq!(
        expected.prompt_tokens, actual.prompt_tokens,
        "{label}: prompt usage"
    );
    assert_eq!(
        expected.completion_tokens, actual.completion_tokens,
        "{label}: completion usage"
    );
    assert_eq!(
        expected.finish_reason, actual.finish_reason,
        "{label}: finish reason"
    );
    assert!(
        (1..=max_tokens).contains(&actual.completion_tokens),
        "{label}: completion count must respect max_tokens={max_tokens}"
    );
}

fn set_reuse(engine: &mut LlamaEngine, enabled: bool) {
    engine.session.with_dependent_mut(|_, session| {
        session.reset();
        session.reuse_prefix = enabled && session.native_state.ordinary_attention;
        assert!(session.cached_prompt.is_empty());
        assert!(!session.native_state.dirty);
    });
}

// Observe the production reset policy against real native memory, without a
// second removal probe that would empty it before the policy gets to run.
struct ObservedMemory<'a, 'model> {
    context: &'a mut LlamaContext<'model>,
    removal: Option<bool>,
    full_clears: usize,
}

impl GenerationMemory for ObservedMemory<'_, '_> {
    fn remove_all(&mut self) -> bool {
        let removed = GenerationMemory::remove_all(self.context);
        self.removal = Some(removed);
        removed
    }

    fn is_empty(&self) -> bool {
        GenerationMemory::is_empty(self.context)
    }

    fn full_clear(&mut self) {
        self.full_clears += 1;
        GenerationMemory::full_clear(self.context);
    }
}

fn observed_reset(engine: &mut LlamaEngine, force_full: bool) {
    engine.session.with_dependent_mut(|_, session| {
        assert!(
            session.native_state.dirty,
            "reset must exercise populated native state"
        );
        if force_full {
            session.native_state.require_full_clear();
        }
        session.cached_prompt.clear();
        let mut memory = ObservedMemory {
            context: &mut session.context,
            removal: None,
            full_clears: 0,
        };
        session.native_state.reset(&mut memory);
        if force_full || !session.native_state.ordinary_attention {
            assert_eq!(memory.removal, None);
            assert_eq!(memory.full_clears, 1);
        } else {
            assert!(
                memory.removal.is_some(),
                "ordinary attention must attempt logical reset"
            );
            if memory.removal == Some(false) {
                assert_eq!(
                    memory.full_clears, 1,
                    "unsupported removal needs full clear"
                );
            }
            // A successful removal that fails the native empty check is also
            // allowed to fall back to full clear; output equality is not waived.
            assert!(
                memory.is_empty(),
                "ordinary-attention reset left native positions"
            );
        }
        eprintln!(
            "reset: forced_full={force_full}, ordinary_attention={}, removal={:?}, full_clears={}",
            session.native_state.ordinary_attention, memory.removal, memory.full_clears
        );
        assert!(!session.native_state.dirty);
        assert!(!session.native_state.full_clear_required);
    });
}

fn check_native_prefix(engine: &LlamaEngine, base: &str, next: &str, label: &str) -> usize {
    engine.session.with_dependent(|loaded, session| {
        if !session.native_state.ordinary_attention {
            assert!(!session.reuse_prefix);
            assert!(session.cached_prompt.is_empty());
            eprintln!("{label}: recurrent/hybrid model; prefix reuse is intentionally disabled");
            return 0;
        }
        assert!(session.reuse_prefix);
        assert_eq!(
            session.cached_prompt,
            tokenize_text_prompt(&loaded.model, base)
        );
        let tokens = tokenize_text_prompt(&loaded.model, next);
        let keep = prefix_to_reuse(&session.cached_prompt, &tokens, session.reuse_prefix);
        assert!(
            keep > 0,
            "{label}: fixture must share a reusable token prefix"
        );
        let min = session.context.kv_cache_seq_pos_min(0);
        let max = session.context.kv_cache_seq_pos_max(0);
        assert!(
            prefix_is_resident(keep, min, max),
            "{label}: prefix {keep} not resident ({min}..={max})"
        );
        eprintln!("{label}: resident reusable prefix={keep}, positions={min}..={max}");
        keep
    })
}

fn check_native_removal(engine: &mut LlamaEngine, keep: usize, label: &str) {
    if keep == 0 {
        return;
    }
    // Probe only AFTER output parity: pre-trimming the old generation suffix
    // before inference would mask bugs in production's own suffix removal.
    engine.session.with_dependent_mut(|_, session| {
        let removed = session.context.clear_kv_cache_seq(Some(0), Some(keep as u32), None);
        match removed {
            Ok(true) => {
                let min = session.context.kv_cache_seq_pos_min(0);
                let max = session.context.kv_cache_seq_pos_max(0);
                assert!(prefix_is_resident(keep, min, max), "{label}: native removal lost the reusable prefix");
                assert_eq!(max, keep as i32 - 1, "{label}: native suffix survived removal");
                eprintln!("{label}: native reusable prefix={keep}, positions={min}..={max}");
            }
            unsupported => {
                eprintln!("{label}: partial removal unsupported ({unsupported:?}); strict output parity still applies");
                // A rejected removal may have partially mutated native state.
                session.native_state.require_full_clear();
                session.reset();
            }
        }
    });
}

fn explicit_native_text_model() -> (PathBuf, Option<String>) {
    let gguf = std::env::var_os("LMA_TEST_TEXT_GGUF");
    let selector = std::env::var_os("LMA_TEST_OLLAMA_MODEL");
    match (gguf, selector) {
        (Some(path), None) => {
            assert!(!path.is_empty(), "LMA_TEST_TEXT_GGUF must not be empty");
            (PathBuf::from(path), None)
        }
        (None, Some(selector)) => {
            let selector = selector
                .into_string()
                .expect("LMA_TEST_OLLAMA_MODEL must be UTF-8");
            assert!(
                !selector.is_empty(),
                "LMA_TEST_OLLAMA_MODEL must not be empty"
            );
            let selector = if selector.starts_with("ollama:") {
                selector
            } else {
                format!("ollama:{selector}")
            };
            let resolved = crate::models::resolve(std::path::Path::new(&selector), None)
                .expect("resolve an already-installed Ollama text model (nothing is downloaded)");
            // Do not load a discovered projector or apply Ollama templates/options.
            (resolved.model, resolved.name)
        }
        (None, None) => panic!("set LMA_TEST_TEXT_GGUF or LMA_TEST_OLLAMA_MODEL explicitly"),
        (Some(_), Some(_)) => {
            panic!("set only one of LMA_TEST_TEXT_GGUF and LMA_TEST_OLLAMA_MODEL")
        }
    }
}

#[test]
#[ignore = "loads one real text model; set LMA_TEST_TEXT_GGUF or LMA_TEST_OLLAMA_MODEL and run alone"]
fn real_native_until_context_matches_exact_finite_budget_and_keeps_model_loaded() {
    let (model, model_id) = explicit_native_text_model();
    let started = Instant::now();
    let timed_out = || started.elapsed() >= Duration::from_secs(120);
    let mut engine = LlamaEngine::load(ServeConfig {
        model,
        model_id,
        projector: None,
        definition: None,
        chat_template: "plain".into(),
        prompt_cache: true,
        gpu_layers: native_test_u32_env("LMA_TEST_TEXT_GPU_LAYERS", 0),
        memory_fallback: "error".into(),
        context: 128,
        batch: 32,
        micro_batch: 32,
        threads: 4,
        flash: "off".into(),
        kv_cache: "f16".into(),
        ..ServeConfig::default()
    })
    .unwrap();
    let prompt = || {
        Prompt::Chat(vec![Message {
            role: "user".into(),
            content: "Continue counting integers, separated by commas: 1, 2, 3, 4,".into(),
            images: vec![],
        }])
    };
    let count = engine.count_prompt(prompt(), &timed_out).unwrap();
    let context = engine
        .session
        .with_dependent(|_, session| session.context.n_ctx().min(engine.config.context));
    let allowance = context.checked_sub(u32::try_from(count).unwrap()).unwrap();
    assert!(
        allowance > 1,
        "fixture needs room beyond the one-token placeholder"
    );
    let request = |until_context, max_tokens| GenerationRequest {
        prompt: prompt(),
        options: Options {
            until_context,
            max_tokens,
            temperature: 0.0,
            top_p: 1.0,
            seed: Some(42),
        },
    };
    let finite = engine
        .generate(request(false, allowance), &timed_out)
        .unwrap();
    let mut streamed = String::new();
    let unlimited = engine
        .generate_stream(request(true, 1), &timed_out, &mut |delta| {
            streamed.push_str(&delta);
            Ok(())
        })
        .unwrap();
    assert_generation_with_limit(
        &finite,
        &unlimited,
        allowance as usize,
        "until context parity",
    );
    assert_eq!(unlimited.prompt_tokens, count);
    assert_eq!(unlimited.text, streamed);
    if unlimited.finish_reason == "length" {
        assert_eq!(unlimited.completion_tokens, allowance as usize);
    } else {
        assert_eq!(unlimited.finish_reason, "stop");
    }
    // The same engine remains usable after reaching EOS/the hard context limit.
    let next = engine.generate(request(false, 1), &timed_out).unwrap();
    assert_eq!(next.completion_tokens, 1);
    assert_eq!(next.prompt_tokens, count);
}

#[test]
#[ignore = "loads one real text model; set LMA_TEST_TEXT_GGUF or LMA_TEST_OLLAMA_MODEL and run alone"]
fn real_native_text_cache_reset_and_greedy_parity() {
    let (model, model_id) = explicit_native_text_model();
    let started = Instant::now();
    // Cooperative deadline, not a hard interrupt: model loading and individual
    // native decode calls cannot be preempted by this cancellation closure.
    let timed_out = || started.elapsed() >= Duration::from_secs(120);
    let mut engine = LlamaEngine::load(ServeConfig {
        model,
        model_id,
        projector: None,
        definition: None,
        chat_template: "plain".into(),
        prompt_cache: true,
        gpu_layers: 0,
        memory_fallback: "mmap".into(),
        context: 512,
        batch: 32,
        micro_batch: 32,
        threads: 4,
        flash: "off".into(),
        kv_cache: "f16".into(),
        ..ServeConfig::default()
    })
    .expect("load the explicitly selected local text model with CPU/pageable mmap");
    assert!(!timed_out(), "120s deadline expired during model loading");
    eprintln!(
        "one CPU/mmap model loaded in {:?}; all timings are diagnostics, not a benchmark",
        started.elapsed()
    );

    let shared = "The small village stood beside a quiet river. Every morning the baker opened the shop and placed fresh bread on the wooden counter.";
    let base = format!("{shared}\nThe next visitor was");
    let extended = format!("{base} a friendly traveler who asked for");
    let changed = format!("{shared}\nThe last visitor bought");
    let shortened =
        "The small village stood beside a quiet river. Every morning the baker opened the shop";
    let cases = [
        ("identical", base.as_str()),
        ("extended", extended.as_str()),
        ("changed suffix", changed.as_str()),
        ("shortened", shortened),
    ];
    let mut clean_base = None;
    for (label, prompt) in cases {
        // One engine/context throughout: reset and toggle the actual Session
        // switch, not config.prompt_cache (which is read only when loading).
        set_reuse(&mut engine, false);
        let _ = generate_text(&mut engine, &base, "cache off: prime base", &timed_out);
        let uncached = generate_text(
            &mut engine,
            prompt,
            &format!("cache off: {label}"),
            &timed_out,
        );
        engine.session.with_dependent(|_, session| {
            assert!(session.cached_prompt.is_empty());
            assert!(!session.native_state.dirty);
        });

        set_reuse(&mut engine, true);
        let _ = generate_text(&mut engine, &base, "cache on: prime base", &timed_out);
        let keep = check_native_prefix(&engine, &base, prompt, label);
        let cached = generate_text(
            &mut engine,
            prompt,
            &format!("cache on: {label}"),
            &timed_out,
        );
        assert_same_generation(&uncached, &cached, label);
        check_native_removal(&mut engine, keep, label);
        if label == "identical" {
            clean_base = Some(uncached);
        }
    }
    let clean_base = clean_base.unwrap();

    // Both resets start with the same freshly primed, dirty native state. Keep
    // it dirty on recurrent/hybrid models too, while leaving prefix reuse off.
    let mut reset_results = Vec::new();
    for force_full in [false, true] {
        set_reuse(&mut engine, true);
        // Session::generate leaves native state dirty even when prefix reuse is
        // disabled; the engine wrapper would immediately reset that state.
        engine.session.with_dependent_mut(|loaded, session| {
            session
                .generate(
                    &loaded.model,
                    &loaded.chat_formatter,
                    &engine.config,
                    text_request(&base),
                    &timed_out,
                    &mut |_| Ok(()),
                )
                .expect("prime real native state for observed reset");
        });
        observed_reset(&mut engine, force_full);
        reset_results.push(generate_text(
            &mut engine,
            &base,
            "after observed reset",
            &timed_out,
        ));
    }
    assert_same_generation(
        &reset_results[0],
        &reset_results[1],
        "logical vs full reset",
    );
    assert_same_generation(&clean_base, &reset_results[0], "reset vs clean prefill");

    // Compare the shortcut and native sampling on each of four successive real
    // logit rows. sample() accepts internally; only the shortcut accepts here.
    engine.session.with_dependent_mut(|loaded, session| {
        session.reset();
        let tokens = tokenize_text_prompt(&loaded.model, &base);
        let mut position = tokens.len() as i32;
        session
            .prefill_text(tokens, 32, &timed_out)
            .expect("prefill real logits");
        let mut native = LlamaSampler::greedy();
        let mut shortcut = generation_sampler(None, &text_request(&base).options);
        for index in 0..4 {
            ensure_active(&timed_out).expect("120s deadline during real-logit parity");
            let token =
                greedy_token(session.context.get_logits_ith(-1)).expect("nonempty real logits");
            shortcut.accept(token);
            assert_eq!(
                token,
                native.sample(&session.context, -1),
                "real greedy token {index}"
            );
            if loaded.model.vocab().is_eog(token) || index == 3 {
                break;
            }
            session.batch.clear();
            session.batch.add(token, position, &[0], true).unwrap();
            session.native_state.mark_dirty();
            session
                .context
                .decode(&mut session.batch)
                .expect("decode greedy parity token");
            position += 1;
        }
        session.reset();
    });

    // Cancel with a populated prompt/native cache, without relying on the model
    // emitting nonempty text (it may immediately select an EOG token).
    set_reuse(&mut engine, true);
    engine.session.with_dependent_mut(|loaded, session| {
        session
            .prefill_text(tokenize_text_prompt(&loaded.model, &base), 32, &timed_out)
            .expect("prime before cancellation");
        assert!(session.native_state.dirty);
    });
    let polls = Cell::new(0);
    let result = engine.generate(text_request(&base), &|| {
        polls.set(polls.get() + 1);
        true
    });
    assert!(polls.get() > 0);
    assert!(
        matches!(result, Err(InferenceError::Internal(ref message)) if message == "request cancelled")
    );
    engine.session.with_dependent(|_, session| {
        assert!(
            session.cached_prompt.is_empty(),
            "cancellation retained a prefix"
        );
        assert!(!session.native_state.dirty);
        assert!(!session.native_state.full_clear_required);
        if session.native_state.ordinary_attention {
            assert!(
                GenerationMemory::is_empty(&session.context),
                "cancellation retained native positions"
            );
        }
    });
    let recovered = generate_text(&mut engine, &base, "after cancellation", &timed_out);
    assert_same_generation(&clean_base, &recovered, "cancellation recovery");
    assert!(!timed_out(), "120s cooperative deadline exceeded");
}

#[derive(Debug)]
struct TraceTensor {
    name: String,
    shape: [i64; 4],
    values: Vec<f32>,
}

static EXPERT_TRACE: std::sync::Mutex<Vec<TraceTensor>> = std::sync::Mutex::new(Vec::new());
static TRACE_LAYER_BOUNDARIES: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub(super) fn trace_context_params(params: LlamaContextParams) -> LlamaContextParams {
    if std::env::var_os("LMA_TEST_EXPERT_CACHE_TRACE").is_some() {
        TRACE_LAYER_BOUNDARIES.store(params.n_ubatch() > 1, std::sync::atomic::Ordering::Relaxed);
        params.with_eval_callback(Some(expert_trace_callback))
    } else {
        params
    }
}

unsafe extern "C" fn expert_trace_callback(
    tensor: *mut llama_cpp_sys_2::ggml_tensor,
    ask: bool,
    _: *mut std::ffi::c_void,
) -> bool {
    // The native observer can run on a worker. Never unwind across the C ABI.
    std::panic::catch_unwind(|| {
        let t = unsafe { &*tensor };
        let name = unsafe { std::ffi::CStr::from_ptr(t.name.as_ptr()) }.to_string_lossy();
        // Streamed prefill has different intermediate topology. Compare layer
        // boundaries rather than aligning a batch tensor with a decode tile.
        if TRACE_LAYER_BOUNDARIES.load(std::sync::atomic::Ordering::Relaxed)
            && ![
                "attn_norm-",
                "ffn_inp-",
                "ffn_norm-",
                "ffn_moe_out-",
                "l_out-",
            ]
            .iter()
            .any(|prefix| name.starts_with(prefix))
        {
            return false;
        }
        let observe = t.type_ == llama_cpp_sys_2::GGML_TYPE_F32
            && unsafe { llama_cpp_sys_2::ggml_is_contiguous(tensor) }
            && [
                "attn_norm-",
                "Qcur-",
                "Kcur-",
                "Vcur-",
                "ffn_inp-",
                "ffn_norm-",
                "ffn_moe_logits-",
                "ffn_moe_weights_norm-",
                "ffn_moe_up-",
                "ffn_moe_gate-",
                "ffn_moe_down-",
                "ffn_moe_out-",
                "l_out-",
            ]
            .iter()
            .any(|prefix| name.starts_with(prefix));
        if ask || !observe {
            return observe;
        }
        let bytes = unsafe { llama_cpp_sys_2::ggml_nbytes(tensor) };
        assert!(bytes <= 16 * 1024 * 1024 && bytes % 4 == 0);
        let mut values = vec![0.0f32; bytes / 4];
        if bytes > 0 {
            unsafe {
                llama_cpp_sys_2::ggml_backend_tensor_get(
                    tensor,
                    values.as_mut_ptr().cast(),
                    0,
                    bytes,
                );
            }
        }
        EXPERT_TRACE.lock().unwrap().push(TraceTensor {
            name: name.into_owned(),
            shape: t.ne,
            values,
        });
        true
    })
    .unwrap_or(false)
}

fn take_expert_trace() -> Vec<TraceTensor> {
    std::mem::take(&mut *EXPERT_TRACE.lock().unwrap())
}

fn compare_expert_traces(expected: &[TraceTensor], actual: &[TraceTensor], label: &str) {
    if expected.is_empty() && actual.is_empty() {
        return;
    }
    assert_eq!(expected.len(), actual.len(), "{label}: trace length");
    let mut by_name: std::collections::HashMap<&str, std::collections::VecDeque<&TraceTensor>> =
        std::collections::HashMap::new();
    for tensor in actual {
        by_name.entry(&tensor.name).or_default().push_back(tensor);
    }
    let mut differences = 0;
    for (index, a) in expected.iter().enumerate() {
        let b = by_name
            .get_mut(a.name.as_str())
            .and_then(|tensors| tensors.pop_front())
            .unwrap_or_else(|| panic!("{label}: missing trace tensor {}", a.name));
        assert_eq!(a.shape, b.shape, "{label}: trace shape {index}");
        let max_abs = a
            .values
            .iter()
            .zip(&b.values)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if max_abs > 0.0 {
            if differences < 12 {
                eprintln!(
                    "{label}: first-divergence trace[{index}] {} {:?} max_abs={max_abs}",
                    a.name, a.shape
                );
            }
            differences += 1;
        }
    }
    eprintln!(
        "{label}: {differences}/{} observed tensors differ (observer changes graph partitions)",
        expected.len()
    );
}

fn native_test_u32_env(name: &str, default: u32) -> u32 {
    match std::env::var(name) {
        Ok(value) => value
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be a nonnegative u32 integer")),
        Err(std::env::VarError::NotPresent) => default,
        Err(error) => panic!("{name}: {error}"),
    }
}

fn expert_cache_generation(
    engine: &mut LlamaEngine,
    prompt: &str,
    max_tokens: u32,
    label: &str,
    cancelled: &dyn Fn() -> bool,
) -> (Generation, Vec<f32>) {
    ensure_active(cancelled).expect("240s deadline before expert-cache request");
    let started = Instant::now();
    let mut request = text_request(prompt);
    request.options.max_tokens = max_tokens;
    // Use the real Session generation path so logits are copied BEFORE the
    // engine wrapper's prompt_cache=false reset invalidates native outputs.
    let result = engine.session.with_dependent_mut(|loaded, session| {
        assert!(
            !session.reuse_prefix,
            "{label}: KV prefix reuse must be off"
        );
        assert!(session.cached_prompt.is_empty());
        assert!(!session.native_state.dirty);
        let generation = session
            .generate(
                &loaded.model,
                &loaded.chat_formatter,
                &engine.config,
                request,
                cancelled,
                &mut |_| Ok(()),
            )
            .unwrap_or_else(|error| panic!("{label}: {error:?}"));
        let logits = session.context.get_logits_ith(-1).to_vec();
        assert!(!logits.is_empty(), "{label}: missing final real logits");
        assert!(session.native_state.dirty, "{label}: no native decode");
        ensure_active(cancelled).expect("240s deadline after expert-cache request");
        // Mirror generate_with_output's no-prefix-retention policy, without
        // discarding the model-owned expert slots between requests.
        session.reset();
        assert!(session.cached_prompt.is_empty());
        assert!(!session.native_state.dirty);
        if session.native_state.ordinary_attention {
            assert!(GenerationMemory::is_empty(&session.context));
        }
        (generation, logits)
    });
    eprintln!("{label}: elapsed={:?} (diagnostic only)", started.elapsed());
    result
}

fn assert_expert_cache_result(
    expected: &(Generation, Vec<f32>),
    actual: &(Generation, Vec<f32>),
    max_tokens: u32,
    gpu_layers: u32,
    label: &str,
) {
    assert_generation_with_limit(&expected.0, &actual.0, max_tokens as usize, label);
    assert_eq!(expected.1.len(), actual.1.len(), "{label}: logit count");
    if gpu_layers > 0 {
        let max_abs = expected
            .1
            .iter()
            .zip(&actual.1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let squared: f64 = expected
            .1
            .iter()
            .zip(&actual.1)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum();
        let rmse = (squared / expected.1.len() as f64).sqrt();
        let reference_rms = (expected
            .1
            .iter()
            .map(|v| f64::from(*v).powi(2))
            .sum::<f64>()
            / expected.1.len() as f64)
            .sqrt()
            .max(1.0);
        let normalized_rmse = rmse / reference_rms;
        eprintln!(
            "{label}: GPU final-logit max_abs={max_abs}, rmse={rmse}, normalized_rmse={normalized_rmse}"
        );
        // Compact and full Q4_K matmuls are checked bitwise in the native Metal
        // test. Complete graphs have different scheduler/fusion boundaries due
        // to CPU admission nodes; require tight vector error AND exact output,
        // rather than asserting bitwise reduction order across those graphs.
        assert!(
            normalized_rmse <= 1e-3,
            "{label}: GPU logit-vector drift {normalized_rmse}"
        );
    }
    for (index, (&expected, &actual)) in expected.1.iter().zip(&actual.1).enumerate() {
        assert!(
            expected.is_finite() && actual.is_finite(),
            "{label}: nonfinite final logit {index}: {expected} vs {actual}"
        );
        if gpu_layers == 0 {
            assert_eq!(expected, actual, "{label}: exact CPU final logit {index}");
        } else {
            // Dense GPU vs CPU expert placement can change floating-point
            // kernels. Only logits get a tolerance; output/usage stay exact.
            let tolerance = 1e-2 + 1e-3 * expected.abs();
            assert!(
                (expected - actual).abs() <= tolerance,
                "{label}: mixed GPU/CPU final logit {index}: {expected} vs {actual}; tolerance=1e-2 + 1e-3*abs(expected)={tolerance}; normalized RMSE is separately bounded at 1e-3"
            );
        }
    }
    eprintln!(
        "{label}: strict raw output/usage/finish parity; final logits: {}",
        if gpu_layers == 0 {
            "CPU exact"
        } else {
            "GPU graph tolerance abs=1e-2, rel=1e-3, normalized RMSE<=1e-3"
        }
    );
}

fn native_expert_cache_stats(
    engine: &LlamaEngine,
    budget_bytes: u64,
    label: &str,
) -> llama_cpp_2::model::ExpertCacheStats {
    let stats = engine
        .session
        .borrow_owner()
        .model
        .expert_cache_stats()
        .unwrap_or_else(|| panic!("{label}: expected native expert-cache stats"));
    assert_eq!(stats.budget_bytes, budget_bytes, "{label}: byte budget");
    assert!(stats.allocated_bytes > 0, "{label}: no compact allocation");
    assert!(
        stats.allocated_bytes <= budget_bytes,
        "{label}: compact allocations exceed budget (not an RSS limit)"
    );
    assert!(
        stats.evictions <= stats.misses,
        "{label}: evictions vs misses"
    );
    // Copying more raw projection bytes than all compact allocations can hold
    // proves slot reuse. Otherwise a large budget/model may never need eviction.
    if stats.bytes_copied > stats.allocated_bytes {
        assert!(
            stats.evictions > 0,
            "{label}: provable eviction not counted"
        );
    }
    eprintln!("{label}: {stats:?} (persistent expert slots, not total RSS)");
    stats
}

#[test]
#[ignore = "loads a supported local MoE twice, sequentially; set LMA_TEST_TEXT_GGUF or LMA_TEST_OLLAMA_MODEL and run alone"]
fn real_native_expert_cache_matches_uncached_and_reuses_slots() {
    let started = Instant::now();
    // Cooperative only: resolution/loading and an individual native decode
    // cannot be preempted. No retry, including on timeout or GPU load failure.
    let timed_out = || started.elapsed() >= Duration::from_secs(240);
    let (model, model_id) = explicit_native_text_model();
    let cache_mib = native_test_u32_env("LMA_TEST_EXPERT_CACHE_MIB", 1024);
    assert!(
        (1..=1_048_576).contains(&cache_mib),
        "LMA_TEST_EXPERT_CACHE_MIB must be positive and at most 1048576; the model's aligned active top-k must fit"
    );
    let budget_bytes = u64::from(cache_mib) * 1_048_576;
    let gpu_layers = native_test_u32_env("LMA_TEST_EXPERT_CACHE_GPU_LAYERS", 0);
    let micro_batch = native_test_u32_env("LMA_TEST_EXPERT_CACHE_MICRO_BATCH", 1);
    assert!(
        (1..=32).contains(&micro_batch),
        "test micro-batch must be 1..=32"
    );
    let config = ServeConfig {
        model,
        model_id,
        projector: None,
        definition: None,
        chat_template: "plain".into(),
        prompt_cache: false,
        gpu_layers,
        memory_fallback: "mmap".into(),
        expert_cache_mib: 0,
        context: 256,
        batch: 32,
        micro_batch,
        threads: 4,
        flash: "off".into(),
        kv_cache: "f16".into(),
        ..ServeConfig::default()
    };
    let load_once = |mut config: ServeConfig| {
        ensure_active(&timed_out).expect("240s deadline before model loading");
        config
            .apply_expert_cache_policy()
            .expect("expert-cache policy");
        validate_gguf_file(&config.model, "model").expect("explicit local GGUF");
        // Bypass load()'s automatic GPU-to-CPU retry: both models must use the
        // requested placement, and the test must never silently load a third.
        let mode = if gpu_layers == 0 {
            LoadMode::CpuMmap
        } else {
            LoadMode::Requested
        };
        // Match arithmetic, not just weight bytes: CPU_REPACK uses a different
        // Q4_K accumulation order from plain CPU mmap experts. Leave production
        // defaults untouched and disable repacking in BOTH test instances.
        let engine = LlamaEngine::load_attempt_with_params(config, KvCacheType::F16, mode,
            |params| params.with_use_extra_bufts(false))
            .unwrap_or_else(|error| {
                panic!(
                    "single native load failed (no retry): {error:?}; cache requires a supported qwen2moe/qwen3moe/mellum layout and sufficient LMA_TEST_EXPERT_CACHE_MIB for aligned active top-k"
                )
            });
        assert!(!timed_out(), "240s deadline expired during model loading");
        assert_eq!(engine.config.gpu_layers, gpu_layers);
        assert_eq!(engine.config.micro_batch, micro_batch);
        engine.session.with_dependent(|_, session| {
            assert_eq!(session.context.n_ubatch(), micro_batch);
        });
        engine
    };

    let mut baseline = load_once(config.clone());
    let loaded = baseline.session.borrow_owner();
    let architecture = loaded
        .model
        .meta_val_str("general.architecture")
        .expect("GGUF architecture");
    assert!(
        matches!(architecture.as_str(), "qwen2moe" | "qwen3moe" | "mellum"),
        "unsupported expert-cache architecture {architecture}; select a supported qwen2moe/qwen3moe/mellum layout"
    );
    assert!(loaded.model.expert_cache_stats().is_none());
    // Include model-added BOS/EOS in this check. A multi-token prompt can
    // legitimately churn minimum top-k slots even when repeated identically.
    let warm_prompt = ["a", "1", "\n", "", "<s>", "<|endoftext|>"]
        .into_iter()
        .find(|prompt| tokenize_text_prompt(&loaded.model, prompt).len() == 1)
        .expect("fixture needs a raw prompt yielding exactly one token including BOS/EOS");
    eprintln!(
        "architecture={architecture}; gpu_layers={gpu_layers}; micro_batch={micro_batch}; cache_mib={cache_mib}; one-token raw prompt={warm_prompt:?}; sequential models only"
    );
    let first_prompt = "The quiet village stood beside a river. Next";
    let mut cases = vec![
        ("short prompt A", first_prompt, 8),
        (
            "short prompt B",
            "A baker placed fresh bread on the counter. Then",
            8,
        ),
        ("repeated prompt A", first_prompt, 8),
        ("one-token prime", warm_prompt, 1),
        ("one-token repeat", warm_prompt, 1),
    ];
    let long_prompt = "The engineer inspected the parser, checked every boundary, and wrote tests before changing the code. ".repeat(4);
    if std::env::var_os("LMA_TEST_EXPERT_CACHE_LONG_PROMPT").is_some() {
        let tokens = tokenize_text_prompt(&loaded.model, &long_prompt).len();
        assert!(
            tokens > 32 && tokens + 8 <= 256,
            "long-prompt fixture: {tokens} tokens"
        );
        eprintln!("long-prompt coverage: {tokens} tokens, micro_batch={micro_batch}");
        cases.push(("multi-microbatch prompt", &long_prompt, 8));
    }
    let mut expected_traces = Vec::new();
    let expected: Vec<_> = cases
        .iter()
        .map(|&(label, prompt, max_tokens)| {
            let result = expert_cache_generation(
                &mut baseline,
                prompt,
                max_tokens,
                &format!("uncached: {label}"),
                &timed_out,
            );
            expected_traces.push(take_expert_trace());
            result
        })
        .collect();
    assert_expert_cache_result(&expected[0], &expected[2], 8, gpu_layers, "uncached repeat");
    drop(baseline); // Release context, model and singleton backend before reload.
    assert!(!timed_out(), "240s deadline before cached model loading");

    let mut cached_config = config;
    cached_config.expert_cache_mib = cache_mib;
    let mut cached = load_once(cached_config);
    let mut before = native_expert_cache_stats(&cached, budget_bytes, "cached: loaded");
    assert_eq!(
        before.gpu,
        gpu_layers > 0,
        "cache must reside on the requested device, not silently fall back"
    );
    assert_eq!(before.hits, 0, "loading must not execute expert routes");
    assert_eq!(before.misses, 0);
    assert!(
        before.bytes_copied <= 64 * 1024 * 1024,
        "startup warmup copy cap"
    );
    let cold_start = before.bytes_copied == 0;
    let mut first_routes = 0;
    for (index, &(label, prompt, max_tokens)) in cases.iter().enumerate() {
        let label = format!("cached: {label}");
        let actual = expert_cache_generation(&mut cached, prompt, max_tokens, &label, &timed_out);
        compare_expert_traces(&expected_traces[index], &take_expert_trace(), &label);
        assert_expert_cache_result(&expected[index], &actual, max_tokens, gpu_layers, &label);
        let after = native_expert_cache_stats(&cached, budget_bytes, &label);
        assert!(after.hits >= before.hits, "{label}: hits decreased");
        assert!(after.misses >= before.misses, "{label}: misses decreased");
        assert!(
            after.bytes_copied >= before.bytes_copied,
            "{label}: copies decreased"
        );
        assert!(
            after.evictions >= before.evictions,
            "{label}: evictions decreased"
        );
        let hits = after.hits - before.hits;
        let misses = after.misses - before.misses;
        assert!(hits + misses > 0, "{label}: native router did not execute");
        if index == 0 {
            if cold_start {
                assert!(misses > 0, "cold first request must populate expert slots");
                assert!(after.bytes_copied > before.bytes_copied);
            }
            first_routes = hits + misses;
        } else if index == 2 {
            // Same tokens imply the same number of top-k selections, NOT zero
            // misses: a minimum-sized cache can evict between prompt tokens.
            assert_eq!(hits + misses, first_routes, "{label}: route count");
            assert!(
                misses <= first_routes,
                "{label}: repeated-request miss bound"
            );
        } else if index == 4 {
            // max_tokens=1 never decodes the sampled completion. With exactly
            // one raw input token and no KV retention, every active expert from
            // prime must still occupy its layer's slots, even at minimum budget.
            assert!(hits > 0, "{label}: repeated decode must hit expert slots");
            assert_eq!(
                after.misses, before.misses,
                "{label}: no source reads needed"
            );
            assert_eq!(
                after.bytes_copied, before.bytes_copied,
                "{label}: hit copied weights"
            );
            assert_eq!(
                after.evictions, before.evictions,
                "{label}: hit evicted slots"
            );
        }
        before = after;
    }
    // All-false output flags are valid: the final layer has an empty routing
    // view, and must neither read empty IDs nor unwind across the C ABI.
    cached.session.with_dependent_mut(|loaded, session| {
        session.reset();
        let tokens = tokenize_text_prompt(&loaded.model, "This prompt requests no logits.");
        assert!(tokens.len() <= 32);
        session.batch.clear();
        for (position, token) in tokens.into_iter().enumerate() {
            session
                .batch
                .add(token, position as i32, &[0], false)
                .unwrap();
        }
        session.native_state.mark_dirty();
        session
            .context
            .decode(&mut session.batch)
            .expect("valid zero-output microbatches");
        session.reset();
    });
    let recovered = expert_cache_generation(
        &mut cached,
        first_prompt,
        8,
        "after zero-output decode",
        &timed_out,
    );
    assert_expert_cache_result(
        &expected[0],
        &recovered,
        8,
        gpu_layers,
        "zero-output recovery",
    );
    let rejected = cached.embed(
        EmbeddingRequest {
            input: vec!["expert cache embedding rejection".into()],
            truncate: false,
        },
        &timed_out,
    );
    assert!(
        matches!(rejected, Err(InferenceError::Invalid(ref message)) if message.contains("expert_cache_mib")),
        "expert-cache embeddings must return Invalid: {rejected:?}"
    );
    cached
        .session
        .with_dependent(|_, session| assert!(session.embedding.is_none()));
    drop(cached);
    assert!(
        !timed_out(),
        "240s cooperative expert-cache deadline exceeded"
    );
    eprintln!(
        "expert-cache native regression finished in {:?}; no retries",
        started.elapsed()
    );
}

#[test]
#[ignore = "loads one installed MoE at a time and writes expert-use history; isolate LMA_EXPERT_CACHE_DIR"]
fn real_native_expert_history_survives_reload_and_warms_frequent_bundles() {
    let (model, model_id) = explicit_native_text_model();
    let config = ServeConfig {
        model,
        model_id,
        expert_cache_mib: native_test_u32_env("LMA_TEST_EXPERT_CACHE_MIB", 1024),
        gpu_layers: native_test_u32_env("LMA_TEST_EXPERT_CACHE_GPU_LAYERS", 0),
        context: 256,
        batch: 32,
        micro_batch: 1,
        threads: 4,
        flash: "off".into(),
        kv_cache: "f16".into(),
        memory_fallback: "mmap".into(),
        chat_template: "plain".into(),
        prompt_cache: false,
        ..ServeConfig::default()
    };
    let started = Instant::now();
    let deadline = || started.elapsed() >= Duration::from_secs(240);
    let gpu_layers = config.gpu_layers;
    let load_once = |config: ServeConfig| {
        assert!(!deadline(), "240s deadline before history model load");
        let mode = if gpu_layers == 0 {
            LoadMode::CpuMmap
        } else {
            LoadMode::Requested
        };
        LlamaEngine::load_attempt_with_params(config, KvCacheType::F16, mode, |params| params)
            .expect("single history model load; no GPU-to-CPU retry")
    };
    let mut first = load_once(config.clone());
    assert_eq!(
        first
            .session
            .borrow_owner()
            .model
            .expert_cache_stats()
            .unwrap()
            .gpu,
        gpu_layers > 0
    );
    assert!(
        first.session.borrow_owner().expert_history.is_some(),
        "history must be available"
    );
    let prompt = "fn main() { println!(\"hello\"); }\n";
    let expected = generate_text(&mut first, prompt, "coding history seed", &deadline);
    let usage = first.session.borrow_owner().model.expert_usage().unwrap();
    assert!(usage.iter().any(|entry| entry.count > 0));
    drop(first); // Flush deltas before destroying model/backend, then reload.
    assert!(!deadline());
    let mut second = load_once(config);
    let loaded = second.session.borrow_owner();
    assert!(loaded.expert_history.is_some());
    assert_eq!(
        loaded.model.expert_usage().unwrap(),
        usage,
        "restored exact per-expert frequencies"
    );
    let stats = loaded.model.expert_cache_stats().unwrap();
    assert_eq!(
        stats.gpu,
        gpu_layers > 0,
        "history warms the requested cache device"
    );
    assert!(
        stats.bytes_copied > 0,
        "frequent bundles must be warm before the first request"
    );
    assert!(stats.bytes_copied <= 64 * 1024 * 1024, "startup copy cap");
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0, "warmup is not a demand miss");
    let actual = generate_text(&mut second, prompt, "coding history warm reload", &deadline);
    assert_same_generation(&expected, &actual, "history does not change output");
    drop(second);
    assert!(!deadline());
    eprintln!("history restored and warmed {stats:?}; output unchanged");
}
