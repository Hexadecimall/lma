//! In-memory n-gram hints for verified speculative greedy decoding.
//!
//! This is not an answer cache or trusted KV state. Prompt tokens may contain
//! untrusted external data, and even previously generated tokens are only draft
//! candidates. The caller must verify every candidate against real native batch
//! logits before accepting or emitting it, and must never record EOS tokens.

use std::{collections::HashMap, ops::Range};

use llama_cpp_2::token::LlamaToken;

const CONTEXT_TOKENS: usize = 4;
const MAX_DRAFT_TOKENS: usize = 7;
const MAX_HISTORY_TOKENS: usize = 32_768;

/// Bounded, session-local token history; nothing is persisted to disk.
///
/// Each remembered prompt/output pair is a separate record. Matches can cross
/// its prompt/output boundary, but never cross between unrelated records.
#[derive(Default)]
pub(super) struct DraftHistory {
    tokens: Vec<LlamaToken>,
    record_ends: Vec<usize>,
    continuations: HashMap<[i32; CONTEXT_TOKENS], Range<usize>>,
}

impl DraftHistory {
    /// Suggest tokens *after* the already greedily selected `first` token.
    ///
    /// Uses the last three tokens of `tail` plus `first` as the key, choosing
    /// the most recent occurrence that has a following token. Returns at most
    /// seven unverified candidates, all requiring native-logit verification.
    pub(super) fn propose(
        &self,
        tail: &[LlamaToken],
        first: LlamaToken,
        max: usize,
    ) -> Vec<LlamaToken> {
        if tail.len() < CONTEXT_TOKENS - 1 || max == 0 {
            return Vec::new();
        }
        let tail = &tail[tail.len() - (CONTEXT_TOKENS - 1)..];
        let key = [tail[0].0, tail[1].0, tail[2].0, first.0];
        let Some(continuation) = self.continuations.get(&key) else {
            return Vec::new();
        };
        let count = max.min(continuation.len());
        self.tokens[continuation.start..continuation.start + count].to_vec()
    }

    /// Remember a prompt and its verified generated tokens, excluding EOS.
    ///
    /// Retains at most 32,768 tokens, evicting the oldest tokens first. Oversized
    /// records retain only their suffix. Rebuilds the index here so proposing
    /// on the decoding hot path does not scan history. Remembered content is
    /// only a hint for future verification, never permission to replay output.
    pub(super) fn remember(&mut self, prompt: &[LlamaToken], generated: &[LlamaToken]) {
        let generated_len = generated.len().min(MAX_HISTORY_TOKENS);
        let prompt_len = prompt.len().min(MAX_HISTORY_TOKENS - generated_len);
        let incoming_len = prompt_len + generated_len;
        if incoming_len == 0 {
            return;
        }

        let remove = self
            .tokens
            .len()
            .saturating_sub(MAX_HISTORY_TOKENS - incoming_len);
        if remove > 0 {
            self.tokens.drain(..remove);
            self.record_ends.retain_mut(|end| {
                if *end <= remove {
                    false
                } else {
                    *end -= remove;
                    true
                }
            });
        }
        self.tokens
            .extend_from_slice(&prompt[prompt.len() - prompt_len..]);
        self.tokens
            .extend_from_slice(&generated[generated.len() - generated_len..]);
        self.record_ends.push(self.tokens.len());

        self.continuations.clear();
        let mut start = 0;
        for &end in &self.record_ends {
            // Five-token windows exclude keys with no following continuation.
            for (offset, window) in self.tokens[start..end]
                .windows(CONTEXT_TOKENS + 1)
                .enumerate()
            {
                let key = [window[0].0, window[1].0, window[2].0, window[3].0];
                let next = start + offset + CONTEXT_TOKENS;
                self.continuations
                    .insert(key, next..end.min(next + MAX_DRAFT_TOKENS));
            }
            start = end;
        }
    }

    /// Forget all history and indexed hints (not a secure memory wipe).
    pub(super) fn clear(&mut self) {
        self.tokens.clear();
        self.record_ends.clear();
        self.continuations.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(ids: &[i32]) -> Vec<LlamaToken> {
        ids.iter().copied().map(LlamaToken).collect()
    }

    #[test]
    fn empty_short_zero_and_missing_contexts_return_no_draft() {
        let mut history = DraftHistory::default();
        let tail = tokens(&[1, 2, 3]);
        assert!(history.propose(&tail, LlamaToken(4), 7).is_empty());
        history.remember(&tokens(&[1, 2, 3, 4]), &tokens(&[5, 6]));
        for len in 0..3 {
            assert!(history.propose(&tail[..len], LlamaToken(4), 7).is_empty());
        }
        assert!(history.propose(&tail, LlamaToken(4), 0).is_empty());
        assert!(history.propose(&tail, LlamaToken(9), 7).is_empty());
        assert!(
            history
                .propose(&tokens(&[9, 2, 3]), LlamaToken(4), 7)
                .is_empty()
        );
    }

    #[test]
    fn uses_last_three_tail_tokens_and_returns_only_following_tokens() {
        let mut history = DraftHistory::default();
        history.remember(&tokens(&[1, 2]), &tokens(&[3, 4, 5, 6]));
        assert_eq!(
            history.propose(&tokens(&[99, 1, 2, 3]), LlamaToken(4), 7),
            tokens(&[5, 6])
        );
    }

    #[test]
    fn most_recent_record_wins_without_crossing_record_boundaries() {
        let mut history = DraftHistory::default();
        history.remember(&tokens(&[1, 2, 3, 4]), &tokens(&[5, 6]));
        history.remember(&tokens(&[1, 2, 3, 4]), &tokens(&[9]));
        assert_eq!(
            history.propose(&tokens(&[1, 2, 3]), LlamaToken(4), 7),
            tokens(&[9])
        );
        assert!(
            history
                .propose(&tokens(&[3, 4, 9]), LlamaToken(1), 7)
                .is_empty()
        );
    }

    #[test]
    fn drafts_and_history_are_bounded_and_can_be_forgotten() {
        let mut history = DraftHistory::default();
        let sequence: Vec<_> = (0..MAX_HISTORY_TOKENS as i32 + 100)
            .map(LlamaToken)
            .collect();
        history.remember(&[], &sequence);
        assert_eq!(history.tokens.len(), MAX_HISTORY_TOKENS);
        assert_eq!(
            history
                .propose(&tokens(&[100, 101, 102]), LlamaToken(103), 100)
                .len(),
            7
        );
        assert_eq!(
            history.propose(&tokens(&[100, 101, 102]), LlamaToken(103), 2),
            tokens(&[104, 105])
        );
        history.clear();
        assert!(history.tokens.is_empty() && history.continuations.is_empty());
    }
}
