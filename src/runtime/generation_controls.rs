use serde_json::Value;

use super::inference::InferenceError;

#[derive(Default, Clone, Debug)]
pub(super) struct GenerationControls {
    pub(super) stop: Vec<String>,
    pub(super) frequency_penalty: f32,
    pub(super) presence_penalty: f32,
    pub(super) logit_bias: Vec<(i32, f32)>,
}

impl GenerationControls {
    pub(super) fn from_openai(body: &Value) -> Result<Self, InferenceError> {
        let mut controls = Self::default();
        match body.get("stop") {
            None | Some(Value::Null) => {}
            Some(Value::String(stop)) => controls.stop.push(stop.clone()),
            Some(Value::Array(stops)) if stops.len() <= 4 => {
                for stop in stops {
                    controls.stop.push(
                        stop.as_str()
                            .ok_or_else(|| invalid("stop must contain only strings"))?
                            .to_owned(),
                    );
                }
            }
            _ => {
                return Err(invalid(
                    "stop must be a string or an array of at most 4 strings",
                ));
            }
        }
        if controls
            .stop
            .iter()
            .any(|stop| stop.is_empty() || stop.len() > 4096 || stop.contains('\0'))
        {
            return Err(invalid(
                "each stop must contain 1 to 4096 UTF-8 bytes and no NUL",
            ));
        }
        controls.frequency_penalty =
            number(body.get("frequency_penalty"), "frequency_penalty", 2.0)?;
        controls.presence_penalty = number(body.get("presence_penalty"), "presence_penalty", 2.0)?;
        match body.get("logit_bias") {
            None | Some(Value::Null) => {}
            Some(Value::Object(biases)) if biases.len() <= 1024 => {
                for (id, bias) in biases {
                    let token = id
                        .parse::<i32>()
                        .ok()
                        .filter(|token| *token >= 0 && token.to_string() == *id)
                        .ok_or_else(|| {
                            invalid("logit_bias keys must be canonical nonnegative i32 token IDs")
                        })?;
                    // Unlike the optional top-level penalties, null entries are invalid.
                    if !bias.is_number() {
                        return Err(invalid(
                            "logit_bias values must be finite numbers between -100 and 100",
                        ));
                    }
                    controls
                        .logit_bias
                        .push((token, number(Some(bias), "logit_bias", 100.0)?));
                }
            }
            _ => {
                return Err(invalid(
                    "logit_bias must be an object with at most 1024 entries",
                ));
            }
        }
        Ok(controls)
    }

    pub(super) fn has_sampling(&self) -> bool {
        self.frequency_penalty != 0.0
            || self.presence_penalty != 0.0
            || self.logit_bias.iter().any(|(_, bias)| *bias != 0.0)
    }

    pub(super) fn is_default(&self) -> bool {
        self.stop.is_empty()
            && self.frequency_penalty == 0.0
            && self.presence_penalty == 0.0
            && self.logit_bias.is_empty()
    }

    pub(super) fn validate_vocab(&self, n_vocab: i32) -> Result<(), InferenceError> {
        if let Some((id, _)) = self
            .logit_bias
            .iter()
            .find(|(id, _)| *id < 0 || *id >= n_vocab)
        {
            return Err(invalid(&format!(
                "logit_bias token ID {id} is outside the loaded vocabulary ({n_vocab} tokens)"
            )));
        }
        Ok(())
    }
}

fn invalid(message: &str) -> InferenceError {
    InferenceError::Invalid(message.to_owned())
}

fn number(value: Option<&Value>, name: &str, limit: f64) -> Result<f32, InferenceError> {
    let value = match value {
        None | Some(Value::Null) => return Ok(0.0),
        Some(value) => value.as_f64(),
    };
    match value {
        Some(value) if value.is_finite() && (-limit..=limit).contains(&value) => Ok(value as f32),
        _ => Err(invalid(&format!(
            "{name} must be a finite number between -{limit} and {limit}"
        ))),
    }
}

/// Only the longest suffix which could still become a stop is withheld. The
/// input is already decoded UTF-8, so neither withholding nor emission splits a
/// character. A match is terminal even when more text arrived in the same token.
pub(super) struct StopFilter {
    stops: Vec<String>,
    pending: String,
    stopped: bool,
}

impl StopFilter {
    pub(super) fn new(stops: Vec<String>) -> Option<Self> {
        if stops.is_empty() {
            None
        } else {
            Some(Self {
                stops,
                pending: String::new(),
                stopped: false,
            })
        }
    }

    pub(super) fn push(
        &mut self,
        text: &str,
        last: bool,
        emit: &mut dyn FnMut(&str) -> Result<(), InferenceError>,
    ) -> Result<bool, InferenceError> {
        if self.stopped {
            return Ok(true);
        }
        self.pending.push_str(text);
        let matched = self
            .stops
            .iter()
            .filter_map(|stop| self.pending.find(stop))
            .min();
        let end = if let Some(index) = matched {
            self.stopped = true;
            index
        } else if last {
            self.pending.len()
        } else {
            let max_stop = self.stops.iter().map(String::len).max().unwrap_or(0);
            self.pending
                .char_indices()
                .find_map(|(index, _)| {
                    let suffix = &self.pending[index..];
                    (suffix.len() < max_stop
                        && self.stops.iter().any(|stop| stop.starts_with(suffix)))
                    .then_some(index)
                })
                .unwrap_or(self.pending.len())
        };
        if end > 0 {
            emit(&self.pending[..end])?;
        }
        if self.stopped || last {
            self.pending.clear();
        } else {
            self.pending.drain(..end);
        }
        Ok(self.stopped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn openai_defaults_nulls_boundaries_and_valid_controls() {
        for body in [
            json!({}),
            json!({"stop": null, "frequency_penalty": null, "presence_penalty": null, "logit_bias": null}),
            json!({"stop": []}),
        ] {
            assert!(GenerationControls::from_openai(&body).unwrap().is_default());
        }
        let controls = GenerationControls::from_openai(&json!({
            "stop": ["end", "🦀"], "frequency_penalty": -2, "presence_penalty": 2,
            "logit_bias": {"0": -100, "2147483647": 100}
        }))
        .unwrap();
        assert_eq!(controls.stop, ["end", "🦀"]);
        assert_eq!(controls.frequency_penalty, -2.0);
        assert_eq!(controls.presence_penalty, 2.0);
        assert!(controls.has_sampling());
        assert!(GenerationControls::from_openai(&json!({"stop": "x".repeat(4096)})).is_ok());
    }

    #[test]
    fn openai_rejects_malformed_and_unbounded_controls() {
        for stop in [
            json!(""),
            json!("a\0b"),
            json!(["a", "b", "c", "d", "e"]),
            json!([null]),
            json!([""]),
            json!(42),
            json!({}),
            json!("é".repeat(2049)),
        ] {
            assert!(GenerationControls::from_openai(&json!({"stop": stop})).is_err());
        }
        for name in ["frequency_penalty", "presence_penalty"] {
            for value in [
                json!(2.00001),
                json!(-2.00001),
                json!("1"),
                json!(true),
                json!([]),
            ] {
                let mut body = json!({});
                body[name] = value;
                assert!(GenerationControls::from_openai(&body).is_err());
            }
        }
        for key in [
            "-1",
            "+1",
            "01",
            "-0",
            " 1",
            "1 ",
            "1.0",
            "2147483648",
            "",
            "١",
        ] {
            let mut bias = serde_json::Map::new();
            bias.insert(key.into(), json!(1));
            assert!(
                GenerationControls::from_openai(&json!({"logit_bias": bias})).is_err(),
                "{key}"
            );
        }
        for bias in [
            json!([]),
            json!(1),
            json!({"0": null}),
            json!({"0": "1"}),
            json!({"0": 100.001}),
            json!({"0": -100.001}),
        ] {
            assert!(GenerationControls::from_openai(&json!({"logit_bias": bias})).is_err());
        }
        let mut bias = serde_json::Map::new();
        for id in 0..1024 {
            bias.insert(id.to_string(), json!(0));
        }
        assert!(GenerationControls::from_openai(&json!({"logit_bias": bias.clone()})).is_ok());
        bias.insert("1024".into(), json!(0));
        assert!(GenerationControls::from_openai(&json!({"logit_bias": bias})).is_err());
    }

    #[test]
    fn vocab_validation_includes_zero_bias_entries() {
        let controls = GenerationControls::from_openai(&json!({"logit_bias": {"3": 0}})).unwrap();
        assert!(controls.validate_vocab(3).is_err());
        assert!(controls.validate_vocab(4).is_ok());
    }

    fn filtered(stops: &[&str], chunks: &[&str], last: bool) -> (String, bool) {
        let mut filter = StopFilter::new(stops.iter().map(|s| (*s).into()).collect()).unwrap();
        let mut output = String::new();
        let mut stopped = false;
        for chunk in chunks {
            stopped = filter
                .push(chunk, false, &mut |s| {
                    output.push_str(s);
                    Ok(())
                })
                .unwrap();
        }
        if last {
            stopped = filter
                .push("", true, &mut |s| {
                    output.push_str(s);
                    Ok(())
                })
                .unwrap();
        }
        (output, stopped)
    }

    #[test]
    fn stops_overlap_split_and_discard_everything_after_first_match() {
        assert_eq!(
            filtered(&["abab"], &["a", "ab", "a", "b", "later"], true),
            ("a".into(), true)
        );
        assert_eq!(
            filtered(&["END"], &["hello E", "N", "D discarded END"], true),
            ("hello ".into(), true)
        );
        assert_eq!(
            filtered(&["later", "END", "ND"], &["ok END later"], true),
            ("ok ".into(), true)
        );
        assert_eq!(
            filtered(&["abcd", "bc"], &["abc", "d"], true),
            ("a".into(), true)
        );
        assert_eq!(
            filtered(&["🦀終"], &["café 🦀", "終hidden"], true),
            ("café ".into(), true)
        );
    }

    #[test]
    fn unmatched_suffix_flushes_only_at_normal_end() {
        assert_eq!(
            filtered(&["END"], &["hello EN"], false),
            ("hello ".into(), false)
        );
        assert_eq!(
            filtered(&["END"], &["hello EN"], true),
            ("hello EN".into(), false)
        );
        assert_eq!(
            filtered(&["END"], &["E", "X", "EN"], true),
            ("EXEN".into(), false)
        );
        assert_eq!(filtered(&["🦀終"], &["🦀"], true), ("🦀".into(), false));
        assert!(StopFilter::new(Vec::new()).is_none());
    }
}
