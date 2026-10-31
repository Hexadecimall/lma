//! Incremental SSE decoding for the trusted, in-process completion routes.
use crate::desktop::Session;
use serde_json::Value;
use tokio::sync::mpsc;

const MAX_EVENT: usize = 1024 * 1024;

#[derive(Default)]
struct Decoder {
    line: Vec<u8>,
    data: String,
}

impl Decoder {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>, String> {
        let mut events = Vec::new();
        for &byte in bytes {
            if byte != b'\n' {
                self.line.push(byte);
                if self.line.len() + self.data.len() > MAX_EVENT {
                    return Err("Engine stream event exceeds 1 MiB".into());
                }
                continue;
            }
            let line = std::str::from_utf8(&self.line)
                .map_err(|_| "Engine stream contains invalid UTF-8")?
                .trim_end_matches('\r');
            if line.is_empty() {
                if !self.data.is_empty() {
                    self.data.pop(); // SSE joins data fields with one newline.
                    events.push(std::mem::take(&mut self.data));
                }
            } else if let Some(data) = line.strip_prefix("data:") {
                self.data.push_str(data.strip_prefix(' ').unwrap_or(data));
                self.data.push('\n');
            }
            self.line.clear();
        }
        Ok(events)
    }

    fn finish(&self) -> Result<(), String> {
        if !self.line.is_empty() || !self.data.is_empty() {
            return Err("Engine stream ended in an incomplete event".into());
        }
        Ok(())
    }
}

#[derive(Default)]
struct Reply {
    finished: bool,
    done: bool,
    has_text: bool,
    usage: Value,
    finish_reason: Option<String>,
}

impl Reply {
    fn accept(&mut self, record: &str) -> Result<Option<String>, String> {
        if self.done {
            return Err("Engine sent data after [DONE]".into());
        }
        if record == "[DONE]" {
            if !self.finished {
                return Err("Engine stream ended without a successful finish reason".into());
            }
            self.done = true;
            return Ok(None);
        }
        let value: Value = serde_json::from_str(record)
            .map_err(|error| format!("Engine stream is not JSON: {error}"))?;
        if let Some(error) = value.get("error") {
            return Err(error["message"]
                .as_str()
                .unwrap_or("Engine stream failed")
                .into());
        }
        if value["usage"].is_object() {
            self.usage = value["usage"].clone();
        }
        let choices = value["choices"]
            .as_array()
            .ok_or("Engine stream has no choices")?;
        if choices.is_empty() {
            return Ok(None); // Optional usage-only record after the finish chunk.
        }
        if self.finished {
            return Err("Engine sent a choice after the finish chunk".into());
        }
        let choice = &choices[0];
        if !choice["delta"]["tool_calls"].is_null() || !choice["delta"]["function_call"].is_null() {
            return Err("Tool responses are not supported in terminal chat".into());
        }
        let text = choice["delta"]["content"]
            .as_str()
            .or_else(|| choice["text"].as_str())
            .unwrap_or("");
        self.has_text |= !text.is_empty();
        if let Some(reason) = choice["finish_reason"].as_str() {
            if !matches!(reason, "stop" | "length") {
                return Err(format!("Engine generation did not complete: {reason}"));
            }
            self.finished = true;
            self.finish_reason = Some(reason.into());
        }
        Ok((!text.is_empty()).then(|| text.to_owned()))
    }

    fn finish(mut self) -> Result<Value, String> {
        if !self.finished || !self.done {
            return Err("Engine stream interrupted before completion".into());
        }
        if !self.has_text {
            return Err(if self.finish_reason.as_deref() == Some("length") {
                            "No output room remains in this context. Model stays loaded; /save then /new, or restart with a larger -c."
                        } else { "Engine response contained no text (the model may have emitted EOS immediately). Model stays loaded." }.into());
        }
        if !self.usage.is_object() {
            self.usage = serde_json::json!({});
        }
        self.usage["finish_reason"] = self.finish_reason.into();
        Ok(self.usage)
    }
}

pub async fn generate(
    session: &Session,
    path: &str,
    body: Vec<u8>,
    chunks: mpsc::Sender<String>,
    context_limit: bool,
) -> Result<Value, String> {
    let mut stream = if context_limit {
        session.stream_until_context(path, body).await?
    } else {
        session.stream(path, body).await?
    };
    if !(200..300).contains(&stream.status) {
        let mut bytes = Vec::new();
        while let Some(chunk) = stream.next_chunk().await? {
            if bytes.len() + chunk.len() > MAX_EVENT {
                return Err(format!(
                    "Engine {}: error response exceeds 1 MiB",
                    stream.status
                ));
            }
            bytes.extend_from_slice(&chunk);
        }
        let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        return Err(format!(
            "Engine {}: {}",
            stream.status,
            value["error"]["message"]
                .as_str()
                .unwrap_or("request failed; /logs has native diagnostics")
        ));
    }
    if stream
        .content_type
        .as_deref()
        .and_then(|s| s.split(';').next())
        != Some("text/event-stream")
    {
        return Err("Engine did not return a streaming completion".into());
    }
    let mut decoder = Decoder::default();
    let mut reply = Reply::default();
    while let Some(chunk) = stream.next_chunk().await? {
        for record in decoder.push(&chunk)? {
            if let Some(text) = reply.accept(&record)? {
                // Bounded backpressure keeps a slow terminal from accumulating an
                // unbounded queue. Dropping this task also drops the response body.
                chunks
                    .send(text)
                    .await
                    .map_err(|_| "Terminal stream receiver closed")?;
            }
        }
    }
    decoder.finish()?;
    reply.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn utf8_crlf_comments_and_multiline_data_survive_every_byte_boundary() {
        let input = ": keepalive\r\nevent: message\r\ndata: {\"choices\":\r\ndata: [{\"delta\":{\"content\":\"界🙂\"}}]}\r\n\r\n";
        let mut decoder = Decoder::default();
        let mut records = Vec::new();
        for byte in input.as_bytes() {
            records.extend(decoder.push(&[*byte]).unwrap());
        }
        decoder.finish().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(
            Reply::default().accept(&records[0]).unwrap().as_deref(),
            Some("界🙂")
        );
    }

    #[test]
    fn chat_and_raw_commit_only_with_finish_and_done_and_preserve_usage() {
        for choice in [json!({"delta":{"content":"hi"}}), json!({"text":"hi"})] {
            let mut reply = Reply::default();
            assert_eq!(
                reply
                    .accept(&json!({"choices":[choice]}).to_string())
                    .unwrap()
                    .as_deref(),
                Some("hi")
            );
            reply
                .accept(r#"{"choices":[{"finish_reason":"length"}]}"#)
                .unwrap();
            reply
                .accept(r#"{"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1}}"#)
                .unwrap();
            reply.accept("[DONE]").unwrap();
            assert_eq!(reply.finish().unwrap()["completion_tokens"], 1);
        }
        let mut reply = Reply::default();
        assert!(reply.accept("[DONE]").is_err());
        assert!(Reply::default().finish().is_err());
        let mut reply = Reply::default();
        reply.accept(r#"{"choices":[{"text":"partial"}]}"#).unwrap();
        assert!(reply.finish().is_err());
    }

    #[test]
    fn errors_invalid_json_tool_output_and_unfinished_events_never_succeed() {
        for record in [
            "not json",
            r#"{"error":{"message":"cancelled"}}"#,
            r#"{"choices":[{"finish_reason":"cancelled"}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[]}}]}"#,
        ] {
            assert!(Reply::default().accept(record).is_err());
        }
        let mut decoder = Decoder::default();
        decoder.push(b"data: partial\n").unwrap();
        assert!(decoder.finish().is_err());
        assert!(Decoder::default().push(&vec![b'x'; MAX_EVENT + 1]).is_err());
    }

    #[tokio::test]
    async fn engine_errors_are_preserved_without_emitting_text() {
        use axum::{Json, Router, http::StatusCode, routing::post};
        let router = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error":{"message":"prompt plus output exceeds context"}})),
                )
            }),
        );
        let session = Session::new(router, tokio::spawn(async {}));
        let (chunks, mut output) = mpsc::channel(4);
        let error = generate(
            &session,
            "/v1/chat/completions",
            b"{}".to_vec(),
            chunks,
            false,
        )
        .await
        .unwrap_err();
        assert!(error.contains("Engine 400: prompt plus output exceeds context"));
        assert!(output.recv().await.is_none());
        session.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn complete_streams_return_usage_for_chat_and_raw_without_losing_final_text() {
        use axum::{Router, http::header, routing::post};
        for (path, choice) in [
            ("/v1/chat/completions", json!({"delta":{"content":"hello"}})),
            ("/v1/completions", json!({"text":"hello"})),
        ] {
            let wire = format!(
                "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
                json!({"choices":[choice]}),
                json!({"choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1}})
            );
            let router = Router::new().route(
                path,
                post(move || {
                    let wire = wire.clone();
                    async move {
                        (
                            [(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")],
                            wire,
                        )
                    }
                }),
            );
            let session = Session::new(router, tokio::spawn(async {}));
            let (chunks, mut output) = mpsc::channel(4);
            let usage = generate(&session, path, b"{}".to_vec(), chunks, false)
                .await
                .unwrap();
            assert_eq!(output.recv().await.as_deref(), Some("hello"));
            assert!(output.recv().await.is_none());
            assert_eq!(usage["completion_tokens"], 1);
            session.shutdown().await.unwrap();
        }
    }

    #[tokio::test]
    async fn first_delta_arrives_before_the_body_finishes_and_abort_drops_it() {
        use axum::{Router, body::Body, http::header, routing::post};
        use std::sync::{Arc, Mutex};
        let (send, receive) =
            mpsc::channel::<Result<axum::body::Bytes, std::convert::Infallible>>(4);
        let receive = Arc::new(Mutex::new(Some(receive)));
        let router = Router::new().route(
            "/v1/completions",
            post(move || {
                let receive = receive.lock().unwrap().take().unwrap();
                async move {
                    (
                        [(header::CONTENT_TYPE, "text/event-stream")],
                        Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(receive)),
                    )
                }
            }),
        );
        let session = Arc::new(Session::new(router, tokio::spawn(async {})));
        let borrowed = session.clone();
        let (chunks, mut output) = mpsc::channel(4);
        let task = tokio::spawn(async move {
            generate(&borrowed, "/v1/completions", b"{}".to_vec(), chunks, false).await
        });
        send.send(Ok(axum::body::Bytes::from_static(
            b"data: {\"choices\":[{\"text\":\"first\"}]}\n\n",
        )))
        .await
        .unwrap();
        assert_eq!(
            tokio::time::timeout(std::time::Duration::from_secs(5), output.recv())
                .await
                .expect("first delta was buffered instead of streamed")
                .as_deref(),
            Some("first"),
        );
        assert!(
            !task.is_finished(),
            "The first delta must not wait for the final frame"
        );
        task.abort();
        let _ = task.await;
        assert!(send.is_closed(), "Abort must drop the native response body");
        Arc::try_unwrap(session)
            .ok()
            .unwrap()
            .shutdown()
            .await
            .unwrap();
    }
}
