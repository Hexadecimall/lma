# Local OpenAI Files API — implementation and recovery notes

The Files implementation is in `src/runtime/openai_files.rs`. This document was
reconstructed from the implementation agent's retained records after the original
new documentation disappeared; exact original wording was not recoverable.

## Integration contract

`router<S>(&ServeConfig) -> Router<S>` supplies its own state. The CLI managed
host merges it only for `all`/`openai`, before the JSON generation fallback.
Embedded desktop/TUI inference routers do not expose these storage operations.
The module independently enforces Bearer authentication. Non-loopback binds
without an API key deny all Files access; file IDs are not authorization tokens.
The current authentication model is one local administrator, not multiple tenants.

Routes: multipart `POST /v1/files`, paginated `GET /v1/files`, metadata
`GET /v1/files/{file_id}`, `DELETE /v1/files/{file_id}`, and binary
`GET /v1/files/{file_id}/content`, with bodyless HEAD where appropriate. File
operations never initialize inference, download models, or execute uploaded data.
Upload status is `uploaded`, not a claim of training, indexing or processing.

## Storage contract

`FileStore::new(root)` and `FileStore::open_default()` do not create directories.
Default storage is `~/.lma/openai/files`. The core exposes asynchronous
`metadata(id)`, `open_content(id)`, bounded `read(id, max_bytes)`,
`store_bytes(filename, purpose, bytes, expires_after)`, `list(ListOptions)`, and
`delete(id)` methods. Metadata includes ID, bytes, timestamps, filename, purpose
and upload status. Listing options are purpose, after, limit and order.
`store_bytes` also accepts `batch_output` for future service-generated outputs;
it is not a public streaming-writer interface.

Bounds are 512 MiB/file, 2 GiB/store, 10,000 files, and a request-body cap of
512 MiB plus 64 KiB. A streaming framing guard limits preambles/epilogues to 4 KiB, each
part's headers to 16 KiB, boundaries to 70 printable ASCII bytes, and forms to
eight parts before the multipart parser can accumulate them. It recognizes
boundaries across chunks without buffering file data. Uploads use chunked I/O
with a 30-second input-idle timeout and 10-minute overall deadline; timeout
returns HTTP 408 `file_upload_timeout`, dropping pending staging and its lock.
An in-flight non-preemptible filesystem operation can still finish after HTTP
cancellation. Purposes and expiration are validated; batch uploads
expire after 30 days by default. Expired files disappear from reads/listings and
are reclaimed on subsequent writes. Mutation locks return 429 on contention.
Private permissions and regular-file/no-follow checks protect owned files.
Publication and marker restoration require atomic no-replace rename on
Linux/Android/macOS and a filesystem that supports it. There is no hard-link or
replace-existing fallback: such fallbacks could leave crash states that violate
the single-link invariant. Windows and FreeBSD Files storage are not implemented
and fail closed; that limitation does not disable their other inference APIs.

## Recovery and durability boundaries

Metadata is the visibility commit marker. Readers and mutations share a
cross-process lock, but two-file publication is not a power-loss transaction.
Metadata staging is retained before publishing bytes. If metadata publication
fails, rollback removes only that upload's newly published data; if unlink fails,
both the data and recovery metadata remain and the API reports
`file_storage_recovery_required`. Deletion similarly retains its withdrawn
metadata until unlink succeeds or the commit marker is restored.

Before the next upload or deletion, bounded staging recovery restores valid
metadata when its data still exists, completing interrupted publication or
rolling back interrupted deletion. Journals without published data are reclaimed.
Conflicting commit markers, malformed journals, unsafe entries, or size mismatches
fail closed and preserve recoverable versions rather than overwrite them. A
conflict needs administrator inspection of the private Files directory; repeated
requests do not discard it. Readers may report missing/error until a mutation
performs recovery. An interrupted upload can therefore become visible on a later
mutation even if its original client never received a successful response.

The rollback/tombstone paths have temporary-root regressions, including injected
unlink failure and metadata conflicts, and the CLI host has profile/auth/lazy-load
tests. Files are synced, but directory synchronization and full power-loss
recovery are not guaranteed. The two-file operation is not a database transaction.
Tests do not establish filesystem durability on every supported operating system.
