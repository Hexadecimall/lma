# Working-tree recovery

The owner authorizes best-effort restoration of unexpectedly missing or damaged
work. Recover automatically rather than stopping only because a file disappeared.

- Snapshot the current affected files before recovery. Keep all versions when
  there is a conflict; never discard unrelated or untracked work.
- Prefer the latest recoverable uncommitted implementation, including retained
  agent/tool records, over reverting to an older committed copy.
- Do not use blanket reset, clean, checkout, or load/stash operations to restore
  the repository. Restore only affected paths, then compile and test.
- State what was recovered and distinguish exact restoration from reconstruction.
  Do not claim unknown or irrecoverable contents were restored byte-for-byte.
- Use `ghx`, not `git`, for version-control operations. Do not commit another
  contributor's unrelated work as part of recovery.
