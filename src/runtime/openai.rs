//! Shared validation for real local text features; unsupported cloud work is never faked.
use serde_json::{Value, json};

use super::{
    generation_controls::GenerationControls,
    inference::{Generation, Prompt},
    server::ApiError,
    tools::{self, bounded_text, fields, string},
};

pub(super) fn controls(body: &Value) -> Result<GenerationControls, ApiError> {
    GenerationControls::from_openai(body).map_err(Into::into)
}

pub(super) fn single_choice(body: &Value, chat: bool) -> Result<(), ApiError> {
    for key in if chat {
        &["n"][..]
    } else {
        &["n", "best_of"][..]
    } {
        if body
            .get(key)
            .is_some_and(|v| !v.is_null() && v.as_u64() != Some(1))
        {
            return Err(ApiError::invalid(format!(
                "{key} currently supports only 1; multiple candidate generation is not implemented"
            )));
        }
    }
    if let Some(value) = body.get("logprobs").filter(|v| !v.is_null())
        && !(chat && value == false)
    {
        return Err(ApiError::invalid(
            "token log probabilities are not implemented; omit logprobs or use null (false for Chat)",
        ));
    }
    if body
        .get("top_logprobs")
        .is_some_and(|v| !v.is_null() && v.as_u64() != Some(0))
    {
        return Err(ApiError::invalid(
            "top_logprobs requires token log-probability support; only null/0 is accepted",
        ));
    }
    if let Some(user) = body.get("user").filter(|v| !v.is_null()) {
        bounded_text(string(user)?, 512)?;
    }
    if let Some(modalities) = body.get("modalities").filter(|v| !v.is_null())
        && modalities != &json!(["text"])
    {
        return Err(ApiError::invalid("only text output modality is supported"));
    }
    if body.get("audio").is_some_and(|v| !v.is_null()) {
        return Err(ApiError::invalid(
            "OpenAI audio output is not implemented on text generation routes",
        ));
    }
    Ok(())
}

pub(super) fn stream_usage(body: &Value, stream: bool) -> Result<bool, ApiError> {
    let Some(options) = body.get("stream_options").filter(|v| !v.is_null()) else {
        return Ok(false);
    };
    fields(options, &["include_usage", "include_obfuscation"])?;
    let boolean = |key| {
        options
            .get(key)
            .map(|v| {
                v.as_bool()
                    .ok_or_else(|| ApiError::invalid("stream options must be booleans"))
            })
            .unwrap_or(Ok(false))
    };
    if !stream || boolean("include_obfuscation")? {
        return Err(ApiError::invalid(
            "stream_options requires stream:true; obfuscation is not implemented",
        ));
    }
    boolean("include_usage")
}

pub(super) fn response_schema(value: Option<&Value>) -> Result<Option<Value>, ApiError> {
    let Some(value) = value.filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let kind = string(&value["type"])?;
    let schema = match kind {
        "text" => {
            fields(value, &["type"])?;
            return Ok(None);
        }
        "json_object" => {
            fields(value, &["type"])?;
            json!({"type":"object"})
        }
        "json_schema" => {
            fields(value, &["type", "json_schema"])?;
            let descriptor = &value["json_schema"];
            fields(descriptor, &["name", "description", "strict", "schema"])?;
            let name = string(&descriptor["name"])?;
            if name.is_empty()
                || name.len() > 64
                || !name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
            {
                return Err(ApiError::invalid(
                    "json_schema.name must be 1..64 letters, digits, underscores or hyphens",
                ));
            }
            if let Some(description) = descriptor.get("description") {
                bounded_text(string(description)?, 1024)?;
            }
            if descriptor.get("strict").is_some_and(|v| !v.is_boolean()) {
                return Err(ApiError::invalid("json_schema.strict must be boolean"));
            }
            descriptor
                .get("schema")
                .filter(|v| v.is_object())
                .cloned()
                .ok_or_else(|| ApiError::invalid("json_schema.schema must be an object"))?
        }
        _ => {
            return Err(ApiError::invalid(
                "response_format.type must be text, json_object or json_schema",
            ));
        }
    };
    tools::validate_output_schema(&schema)?;
    Ok(Some(schema))
}

pub(super) fn constrain(prompt: Prompt, schema: Option<Value>) -> Prompt {
    match schema {
        Some(schema) => Prompt::Constrained {
            prompt: Box::new(prompt),
            schema,
        },
        None => prompt,
    }
}

pub(super) fn validate_output(result: &Generation, schema: Option<&Value>) -> Result<(), ApiError> {
    // Length exhaustion may legitimately produce a partial JSON document.
    if result.finish_reason != "length"
        && let Some(schema) = schema
    {
        let valid = serde_json::from_str::<Value>(&result.text)
            .ok()
            .is_some_and(|value| tools::output_matches(schema, &value));
        if !valid {
            return Err(ApiError::new(
                axum::http::StatusCode::BAD_GATEWAY,
                "invalid_structured_output",
                "model output did not satisfy the requested JSON schema",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn schemas_are_real_bounded_constraints_and_not_ignored_hints() {
        assert!(
            response_schema(Some(&json!({"type":"json_object"})))
                .unwrap()
                .is_some()
        );
        let schema = response_schema(Some(&json!({"type":"json_schema","json_schema":{"name":"Edit","strict":true,"schema":{"type":"object","properties":{"code":{"type":"string"}},"required":["code"],"additionalProperties":false}}}))).unwrap().unwrap();
        let result = |text: &str, finish_reason| Generation {
            text: text.into(),
            prompt_tokens: 1,
            completion_tokens: 1,
            finish_reason,
        };
        assert!(validate_output(&result("{\"code\":\"x\"}", "stop"), Some(&schema)).is_ok());
        assert!(validate_output(&result("{\"wrong\":true}", "stop"), Some(&schema)).is_err());
        assert!(validate_output(&result("{\"code\":", "length"), Some(&schema)).is_ok());
        for value in [
            json!({"type":"json_schema","json_schema":{"name":"x","schema":{"$ref":"remote"}}}),
            json!({"type":"json_object","ignored":true}),
            json!({"type":"xml"}),
        ] {
            assert!(response_schema(Some(&value)).is_err());
        }
    }
    #[test]
    fn sdk_neutral_fields_are_accepted_without_fabricating_features() {
        single_choice(
            &json!({"n":1,"logprobs":false,"top_logprobs":0,"modalities":["text"],"audio":null}),
            true,
        )
        .unwrap();
        single_choice(&json!({"n":1,"best_of":1,"logprobs":null}), false).unwrap();
        for body in [
            json!({"n":2}),
            json!({"logprobs":true}),
            json!({"modalities":["audio"]}),
        ] {
            assert!(single_choice(&body, true).is_err());
        }
        assert!(single_choice(&json!({"logprobs":0}), false).is_err());
        assert!(stream_usage(&json!({"stream_options":{"include_usage":true}}), true).unwrap());
        assert!(stream_usage(&json!({"stream_options":{"include_usage":true}}), false).is_err());
    }
}
