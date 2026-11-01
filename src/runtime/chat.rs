//! Chat formatting is a model-level policy, shared by every frontend and token counting.
use std::sync::atomic::{AtomicBool, Ordering};

use llama_cpp_2::model::{LlamaChatMessage, LlamaChatTemplate, LlamaModel};

use super::inference::{InferenceError, Message, Prompt};
use crate::models::modelfile::Definition;

/// Called once by the common generation/count formatting path, never per token.
/// Request-system provenance keeps tool-generated instructions from suppressing
/// a stored SYSTEM, without trying to recognize protocol text heuristically.
pub(super) fn profile_prompt(prompt: Prompt, definition: Option<&Definition>) -> Prompt {
    fn apply(prompt: Prompt, definition: Option<&Definition>, explicit: Option<bool>) -> Prompt {
        match prompt {
            Prompt::RequestSystem { prompt, explicit } => {
                apply(*prompt, definition, Some(explicit))
            }
            Prompt::Controls { prompt, controls } => Prompt::Controls {
                prompt: Box::new(apply(*prompt, definition, explicit)),
                controls,
            },
            Prompt::Constrained { prompt, schema } => Prompt::Constrained {
                prompt: Box::new(apply(*prompt, definition, explicit)),
                schema,
            },
            Prompt::Text(text) => Prompt::Text(text),
            Prompt::Chat(mut messages) => {
                let Some(definition) = definition else {
                    return Prompt::Chat(messages);
                };
                let has_system = explicit
                    .unwrap_or_else(|| messages.iter().any(|message| message.role == "system"));
                let leading_systems = messages
                    .iter()
                    .take_while(|message| message.role == "system")
                    .count();
                let mut profiled =
                    Vec::with_capacity(messages.len() + definition.messages.len() + 1);
                if leading_systems > 0 {
                    profiled.extend(messages.drain(..leading_systems));
                    // With no client system, leading system text is the tool
                    // adapter's instruction. Merge just as tool preparation
                    // would have merged an original system message.
                    if !has_system && let Some(system) = &definition.system {
                        profiled[0].content.push_str("\n\nSystem instructions:\n");
                        profiled[0].content.push_str(system);
                    }
                } else if !has_system && let Some(system) = &definition.system {
                    profiled.push(Message {
                        role: "system".into(),
                        content: system.clone(),
                        images: Vec::new(),
                    });
                }
                profiled.extend(definition.messages.iter().map(|message| Message {
                    role: message.role.clone(),
                    content: message.content.clone(),
                    images: Vec::new(),
                }));
                profiled.extend(messages);
                Prompt::Chat(profiled)
            }
        }
    }
    apply(prompt, definition, None)
}

pub(super) struct Formatter {
    primary: Option<LlamaChatTemplate>,
    fallback: Fallback,
    automatic: bool,
    plain_override: bool,
    gemma4: bool,
    warned: AtomicBool,
}

enum Fallback {
    Plain,
    Gemma4,
    Native(LlamaChatTemplate),
}

impl Formatter {
    pub(super) fn load(model: &LlamaModel, choice: &str) -> Result<Self, String> {
        let vocab = model.vocab();
        Self::select(choice, model.chat_template(None).ok(), |spelling| {
            // Merely having these strings as ordinary text is not evidence of a
            // trained chat format. Require a single exact special/control token.
            let tokens = vocab.tokenize(spelling.as_bytes(), false, true);
            tokens.len() == 1
                && vocab.is_control(tokens[0])
                && vocab
                    .text(tokens[0])
                    .is_some_and(|text| text.to_bytes() == spelling.as_bytes())
        })
    }

    fn select(
        choice: &str,
        embedded: Option<LlamaChatTemplate>,
        has_special: impl Fn(&str) -> bool,
    ) -> Result<Self, String> {
        if !crate::config::CHAT_TEMPLATES.contains(&choice) {
            return Err(format!(
                "unknown chat template '{choice}'; supported choices: {}",
                crate::config::CHAT_TEMPLATES.join(", ")
            ));
        }
        let automatic = choice == "auto";
        let gemma4 = choice == "gemma4"
            || (matches!(choice, "auto" | "native")
                && embedded.as_ref().and_then(|t| t.to_str().ok())
                    == Some(crate::import::gemma4_chat_template()));
        let primary = match choice {
            "auto" | "native" => embedded,
            "plain" => None,
            name => Some(LlamaChatTemplate::new(name).map_err(|error| error.to_string())?),
        };
        let fallback = match inferred_format(has_special) {
            Some("gemma4") if automatic => Fallback::Gemma4,
            Some(name) if automatic => {
                Fallback::Native(LlamaChatTemplate::new(name).expect("static name"))
            }
            _ => Fallback::Plain,
        };
        Ok(Self {
            primary,
            fallback,
            automatic,
            plain_override: choice == "plain",
            gemma4,
            warned: AtomicBool::new(false),
        })
    }

    pub(super) fn format(
        &self,
        model: &LlamaModel,
        messages: &[Message],
    ) -> Result<String, InferenceError> {
        self.format_with(messages, |template, messages| {
            model
                .apply_chat_template(template, messages, true)
                .map_err(|error| error.to_string())
        })
    }

    fn format_with(
        &self,
        messages: &[Message],
        mut apply: impl FnMut(&LlamaChatTemplate, &[LlamaChatMessage]) -> Result<String, String>,
    ) -> Result<String, InferenceError> {
        if messages.is_empty() {
            return Err(InferenceError::Invalid(
                "chat needs at least one message".into(),
            ));
        }
        // Validate before choosing a formatter: fallback must not hide malformed input.
        let native = messages
            .iter()
            .map(|message| {
                if !matches!(message.role.as_str(), "system" | "user" | "assistant") {
                    return Err(InferenceError::Invalid(
                        "unsupported internal chat role".into(),
                    ));
                }
                LlamaChatMessage::new(message.role.clone(), message.content.clone()).map_err(
                    |error| InferenceError::Invalid(format!("invalid chat message: {error}")),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        if self.gemma4 {
            // The pinned native heuristic renderer does not implement this exact
            // unified protocol. Tokenization supplies the model-configured BOS.
            return Ok(unified_gemma4(messages));
        }
        if let Some(primary) = &self.primary {
            match apply(primary, &native) {
                Ok(text) if !text.is_empty() => return Ok(text),
                Ok(_) if !self.automatic => {
                    return Err(InferenceError::Invalid(
                        "selected chat template produced an empty prompt".into(),
                    ));
                }
                Err(error) if !self.automatic => {
                    return Err(InferenceError::Invalid(format!(
                        "cannot apply selected chat template: {error}; try -ce auto or -ce plain"
                    )));
                }
                _ => {}
            }
        } else if !self.automatic {
            // An explicit plain override is useful for base/completion models.
            // Native-only is strict so users can detect a bad model export.
            return if self.plain_override {
                Ok(plain(messages))
            } else {
                Err(InferenceError::Invalid(
                    "model has no embedded chat template; use -ce auto or -ce plain".into(),
                ))
            };
        }
        let (text, format) = match &self.fallback {
            Fallback::Gemma4 => (unified_gemma4(messages), "gemma4"),
            Fallback::Native(template) => match apply(template, &native) {
                Ok(text) if !text.is_empty() => (text, template.to_str().unwrap_or("inferred")),
                _ => (plain(messages), "plain"),
            },
            Fallback::Plain => (plain(messages), "plain"),
        };
        if !self.warned.swap(true, Ordering::Relaxed) {
            eprintln!(
                "WARNING: embedded chat template missing or unusable; using {format} compatibility formatting. Base/edit models may not follow chat or tool instructions. Override with -ce NAME if needed."
            );
        }
        Ok(text)
    }
}

fn inferred_format(has: impl Fn(&str) -> bool) -> Option<&'static str> {
    [
        (
            "llama3",
            &["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"][..],
        ),
        ("gemma4", &["<|turn>", "<turn|>"][..]),
        ("gemma", &["<start_of_turn>", "<end_of_turn>"][..]),
        ("phi4", &["<|im_start|>", "<|im_sep|>", "<|im_end|>"][..]),
        ("chatml", &["<|im_start|>", "<|im_end|>"][..]),
        ("phi3", &["<|user|>", "<|assistant|>", "<|end|>"][..]),
    ]
    .into_iter()
    .find(|(_, tokens)| tokens.iter().all(|token| has(token)))
    .map(|(name, _)| name)
}

fn unified_gemma4(messages: &[Message]) -> String {
    let mut text = String::new();
    for message in messages {
        text.push_str("<|turn>");
        text.push_str(if message.role == "assistant" {
            "model"
        } else {
            &message.role
        });
        text.push('\n');
        text.push_str(&message.content);
        text.push_str("<turn|>\n");
    }
    // Ordinary chat should not spend its output budget emitting thought markers.
    text.push_str("<|turn>model\n<|channel>thought\n<channel|>");
    text
}

fn plain(messages: &[Message]) -> String {
    let mut text = String::new();
    for message in messages {
        let role = match message.role.as_str() {
            "system" => "System",
            "user" => "User",
            _ => "Assistant",
        };
        text.push_str(role);
        text.push_str(": ");
        text.push_str(&message.content);
        text.push_str("\n\n");
    }
    text.push_str("Assistant:");
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(role: &str, content: &str) -> Message {
        Message {
            role: role.into(),
            content: content.into(),
            images: Vec::new(),
        }
    }
    fn conversation() -> Vec<Message> {
        vec![
            message("system", "Be concise."),
            message("user", "First"),
            message("assistant", "Reply"),
            message("user", "Second 🦀"),
        ]
    }
    fn no_native(_: &LlamaChatTemplate, _: &[LlamaChatMessage]) -> Result<String, String> {
        panic!("plain formatting must not call the native template renderer")
    }

    fn definition() -> Definition {
        Definition {
            from: "base".into(),
            system: Some("Stored system".into()),
            messages: vec![
                crate::models::modelfile::DefinitionMessage {
                    role: "user".into(),
                    content: "Saved question".into(),
                },
                crate::models::modelfile::DefinitionMessage {
                    role: "assistant".into(),
                    content: "Saved answer".into(),
                },
            ],
            ..Default::default()
        }
    }

    fn profiled_messages(prompt: Prompt, definition: &Definition) -> Vec<Message> {
        let Prompt::Chat(messages) = profile_prompt(prompt, Some(definition)) else {
            panic!("expected chat")
        };
        messages
    }

    #[test]
    fn definition_system_and_history_precede_the_request_without_replacing_explicit_system() {
        let definition = definition();
        let messages =
            profiled_messages(Prompt::Chat(vec![message("user", "Current")]), &definition);
        assert_eq!(
            plain(&messages),
            "System: Stored system\n\nUser: Saved question\n\nAssistant: Saved answer\n\nUser: Current\n\nAssistant:"
        );
        let messages = profiled_messages(
            Prompt::Chat(vec![
                message("system", "Explicit"),
                message("user", "Current"),
            ]),
            &definition,
        );
        assert_eq!(
            plain(&messages),
            "System: Explicit\n\nUser: Saved question\n\nAssistant: Saved answer\n\nUser: Current\n\nAssistant:"
        );
        assert!(!plain(&messages).contains("Stored system"));
        assert_eq!(definition.messages.len(), 2);
    }

    #[test]
    fn definition_preserves_request_system_provenance_across_tool_preparation() {
        let definition = definition();
        let prompt = Prompt::Chat(vec![
            message("system", "Tool instructions"),
            message("user", "Current"),
        ])
        .with_request_system(false);
        let messages = profiled_messages(prompt, &definition);
        assert_eq!(
            messages[0].content,
            "Tool instructions\n\nSystem instructions:\nStored system"
        );
        assert_eq!(messages[1].content, "Saved question");
        assert_eq!(messages[3].content, "Current");
        assert_eq!(plain(&messages).matches("Stored system").count(), 1);
        let prompt = Prompt::Chat(vec![
            message(
                "system",
                "Tool instructions\n\nSystem instructions:\nExplicit",
            ),
            message("user", "Current"),
        ])
        .with_request_system(true);
        let messages = profiled_messages(prompt, &definition);
        assert!(!plain(&messages).contains("Stored system"));
        assert_eq!(
            messages[0].content,
            "Tool instructions\n\nSystem instructions:\nExplicit"
        );
        // An explicitly empty Ollama system suppresses SYSTEM without adding an
        // empty message that existing chat validation would reject.
        let messages = profiled_messages(
            Prompt::Chat(vec![message("user", "Current")]).with_request_system(true),
            &definition,
        );
        assert_eq!(messages[0].content, "Saved question");
        assert!(!plain(&messages).contains("Stored system"));
    }

    #[test]
    fn definition_raw_text_and_nested_constraints_remain_intact() {
        let definition = definition();
        let schema = serde_json::json!({"type":"object"});
        let prompt = Prompt::Constrained {
            prompt: Box::new(Prompt::Constrained {
                prompt: Box::new(Prompt::Text("raw <|im_start|> text".into())),
                schema: schema.clone(),
            }),
            schema: schema.clone(),
        }
        .with_request_system(false);
        let Prompt::Constrained {
            prompt,
            schema: outer,
        } = profile_prompt(prompt, Some(&definition))
        else {
            panic!()
        };
        assert_eq!(outer, schema);
        let Prompt::Constrained {
            prompt,
            schema: inner,
        } = *prompt
        else {
            panic!()
        };
        assert_eq!(inner, schema);
        assert!(matches!(*prompt, Prompt::Text(ref text) if text == "raw <|im_start|> text"));
        let messages = profiled_messages(
            Prompt::Chat(vec![
                message("system", "Stored system"),
                message("user", "Current"),
            ]),
            &definition,
        );
        assert_eq!(plain(&messages).matches("Stored system").count(), 1);
        let Prompt::Chat(messages) = profile_prompt(Prompt::Chat(conversation()), None) else {
            panic!()
        };
        assert_eq!(plain(&messages), plain(&conversation()));
    }

    #[test]
    fn imported_gemma4_uses_unified_turns_and_prefills_an_empty_thought_channel() {
        for choice in ["auto", "native", "gemma4"] {
            let formatter = Formatter::select(
                choice,
                Some(LlamaChatTemplate::new(crate::import::gemma4_chat_template()).unwrap()),
                |_| true,
            )
            .unwrap();
            let text = formatter.format_with(&conversation(), no_native).unwrap();
            assert!(text.starts_with("<|turn>system\nBe concise.<turn|>\n"));
            assert!(text.contains("<|turn>model\nReply<turn|>\n"));
            assert!(text.contains("Second 🦀<turn|>\n"));
            assert!(text.ends_with("<|turn>model\n<|channel>thought\n<channel|>"));
            assert!(
                !text.contains("<bos>"),
                "tokenization supplies BOS exactly once"
            );
            assert!(!formatter.warned.load(Ordering::Relaxed));
        }
        let formatter = Formatter::select(
            "plain",
            Some(LlamaChatTemplate::new(crate::import::gemma4_chat_template()).unwrap()),
            |_| true,
        )
        .unwrap();
        assert_eq!(
            formatter.format_with(&conversation(), no_native).unwrap(),
            plain(&conversation())
        );
    }

    #[test]
    fn missing_template_uses_plain_roles_and_preserves_the_entire_conversation() {
        let formatter = Formatter::select("auto", None, |_| false).unwrap();
        let text = formatter.format_with(&conversation(), no_native).unwrap();
        assert_eq!(
            text,
            "System: Be concise.\n\nUser: First\n\nAssistant: Reply\n\nUser: Second 🦀\n\nAssistant:"
        );
        assert!(formatter.warned.load(Ordering::Relaxed));
        assert_eq!(
            text,
            formatter.format_with(&conversation(), no_native).unwrap()
        );
    }

    #[test]
    fn gemma4_special_tokens_provide_a_fallback_for_other_gguf_exports() {
        let formatter = Formatter::select("auto", None, |token| {
            ["<|turn>", "<turn|>"].contains(&token)
        })
        .unwrap();
        assert_eq!(
            formatter.format_with(&conversation(), no_native).unwrap(),
            unified_gemma4(&conversation())
        );
        let formatter = Formatter::select(
            "auto",
            Some(LlamaChatTemplate::new("unsupported-jinja").unwrap()),
            |token| ["<|turn>", "<turn|>"].contains(&token),
        )
        .unwrap();
        assert_eq!(
            formatter
                .format_with(&conversation(), |_, _| Err("unsupported".into()))
                .unwrap(),
            unified_gemma4(&conversation())
        );
    }

    #[test]
    fn usable_embedded_template_wins_over_detected_format() {
        let formatter = Formatter::select(
            "auto",
            Some(LlamaChatTemplate::new("vicuna").unwrap()),
            |_| true,
        )
        .unwrap();
        let text = formatter
            .format_with(&conversation(), |template, messages| {
                assert_eq!(template.to_str().unwrap(), "vicuna");
                assert_eq!(messages.len(), 4);
                Ok("native prompt".into())
            })
            .unwrap();
        assert_eq!(text, "native prompt");
        assert!(!formatter.warned.load(Ordering::Relaxed));
    }

    #[test]
    fn broken_or_empty_embedded_templates_fall_back_without_losing_messages() {
        for empty in [false, true] {
            let formatter = Formatter::select(
                "auto",
                Some(LlamaChatTemplate::new("not-recognized-jinja").unwrap()),
                |_| false,
            )
            .unwrap();
            let text = formatter
                .format_with(&conversation(), |_, _| {
                    if empty {
                        Ok(String::new())
                    } else {
                        Err("unsupported".into())
                    }
                })
                .unwrap();
            assert_eq!(text, plain(&conversation()));
        }
    }

    #[test]
    fn inference_requires_a_complete_signature_and_prefers_specific_formats() {
        assert_eq!(inferred_format(|token| token == "<|im_start|>"), None);
        assert_eq!(
            inferred_format(|token| ["<|im_start|>", "<|im_end|>"].contains(&token)),
            Some("chatml")
        );
        assert_eq!(
            inferred_format(|token| ["<|im_start|>", "<|im_end|>", "<|im_sep|>"].contains(&token)),
            Some("phi4")
        );
        assert_eq!(inferred_format(|_| true), Some("llama3"));
    }

    #[test]
    fn inferred_native_format_can_also_fall_back_to_plain() {
        for supported in [false, true] {
            let formatter = Formatter::select("auto", None, |token| {
                ["<|im_start|>", "<|im_end|>"].contains(&token)
            })
            .unwrap();
            let text = formatter
                .format_with(&conversation(), |template, _| {
                    assert_eq!(template.to_str().unwrap(), "chatml");
                    if supported {
                        Ok("chatml prompt".into())
                    } else {
                        Err("bad render".into())
                    }
                })
                .unwrap();
            assert_eq!(
                text,
                if supported {
                    "chatml prompt".into()
                } else {
                    plain(&conversation())
                }
            );
        }
    }

    #[test]
    fn explicit_overrides_are_strict_and_plain_does_not_apply_a_native_template() {
        let embedded = Some(LlamaChatTemplate::new("llama3").unwrap());
        let formatter = Formatter::select("plain", embedded, |_| true).unwrap();
        assert_eq!(
            formatter.format_with(&conversation(), no_native).unwrap(),
            plain(&conversation())
        );
        let formatter = Formatter::select("native", None, |_| true).unwrap();
        assert!(formatter.format_with(&conversation(), no_native).is_err());
        let formatter = Formatter::select("chatml", None, |_| false).unwrap();
        assert!(
            formatter
                .format_with(&conversation(), |_, _| Err("invalid".into()))
                .is_err()
        );
        assert!(!formatter.warned.load(Ordering::Relaxed));
        assert!(Formatter::select("made-up", None, |_| false).is_err());
    }

    #[test]
    fn fallback_preserves_image_placeholders_for_the_existing_binding_validator() {
        let formatter = Formatter::select("auto", None, |_| false).unwrap();
        let messages = vec![
            message("user", "<__lma_image_0__>猫"),
            message("assistant", "A cat"),
            message("user", "<__lma_image_1__>And this?"),
        ];
        let text = formatter.format_with(&messages, no_native).unwrap();
        assert_eq!(text.matches("<__lma_image_0__>").count(), 1);
        assert_eq!(text.matches("<__lma_image_1__>").count(), 1);
        assert!(text.find("<__lma_image_0__>").unwrap() < text.find("<__lma_image_1__>").unwrap());
    }

    #[test]
    fn fallback_does_not_hide_invalid_roles_nuls_or_empty_conversations() {
        let formatter = Formatter::select("auto", None, |_| false).unwrap();
        assert!(formatter.format_with(&[], no_native).is_err());
        assert!(
            formatter
                .format_with(&[message("unknown", "hello")], no_native)
                .is_err()
        );
        assert!(
            formatter
                .format_with(&[message("user", "bad\0message")], no_native)
                .is_err()
        );
    }
}
