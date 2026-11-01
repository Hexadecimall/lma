# Updating the standalone CLI from releases

```sh
lma u -c                  # Check latest stable: metadata only
lma u                     # Download, verify and update if newer
lma u -v 0.3.4            # Select a stable version; no downgrades
lma u -d                  # Network-free plan; no filesystem changes
# Long syntax belongs in NSS:
lma nss update --check
lma nss update --version v0.3.4
```

The updater uses **only `Hexadecimall/lma` GitHub Releases**, not the current
source branch, Cargo, a toolchain download, a model store or a subprocess updater.
It preserves the compiled OS/architecture/backend. Current official build
minimums are macOS 15 and GNU/Linux glibc 2.39; older/unknown supported-runtime
versions are rejected before replacement rather than installing an unusable ABI. An unavailable CUDA/Metal
asset never silently becomes a CPU build. Stable `MAJOR.MINOR.PATCH` versions,
optionally prefixed with `v`, are supported; drafts/prereleases and downgrades are
rejected. An already-current explicit version is a local no-op.

**Availability:** the authenticated release listing was empty when this feature
was implemented. A newer published release with matching assets must exist before
an update can be installed. The [release workflow](publishing-releases.md) prepares
the exact standalone binary and `SHA256SUMS` contract. Its publication is currently
blocked by the empty root `LICENSE`; the project owner must supply and commit the
correct license. Adding an updater does not publish a release or invent a license.

## Private repository access

This repository is currently private. Provide an appropriate fine-grained token
through `GH_TOKEN` or `GITHUB_TOKEN` with access to this repository and **Contents:
read** permission. `GH_TOKEN` takes precedence when nonempty. Use your environment
or secret manager; do not put a token in command arguments, source, screenshots,
URLs or release files. The updater does not read `ghx`'s credential store or invoke
`ghx`. No token is printed/saved; authentication is sent only to the initial
fixed GitHub API request, never a redirected download host. Public releases work
without a token if repository visibility changes.

A 404 means the release/asset may not exist or private-repository access may be
missing; it is not reported as "successfully up to date". Checks only fetch bounded
release metadata: no executable download, state-directory creation or replacement.
Actual updates contact GitHub and approved HTTPS asset storage, potentially
transferring a large CLI binary. No model weights are fetched.

## Verification and publication

Updates require exactly one matching raw binary asset and a `SHA256SUMS` asset.
The updater bounds metadata, asset sizes, reads and request timeouts; verifies
size and SHA-256; cross-checks GitHub asset digests when provided; and checks the
candidate's native executable header/architecture. It never executes a candidate
to decide whether to trust it. Checksums from the same publisher detect corruption
but are **not an independent signature**, notarization or a defense against a
compromised publishing account.

The current executable is resolved through legitimate CLI symlinks to its actual
regular file. Trusted real parent directories and safe permissions are required;
identity/content are rechecked before publication. State, a cross-process update
lock and retained backups live under **`~/.lma/updates`**. Unix publication stages
beside the destination for an atomic same-filesystem replacement. The previous
binary is backed up first. Failures before publication preserve the installed
executable. A retained backup is for manual recovery, not an automatic downgrade.

No privilege elevation, PATH edits, source-registration rewrite, model changes,
automatic server restart or termination of other processes is performed. Existing
servers continue using their loaded executable; the update takes effect on a new
launch. The CLI updater does not update the desktop companion.

### Windows

A verified executable is staged under `~/.lma/updates` and the command explicitly
reports **NOT INSTALLED**, with quoted PowerShell copy instructions to run after
exiting and after other users of that executable have finished naturally. It does
not claim that a locked running executable was replaced, bypass locks, restart a
server, or launch a hidden helper. Windows publication remains a manual post-exit
step; real Windows running-executable behavior requires platform testing.

### Managed/system and source-build installations

The updater refuses package-manager/system-protected paths, desktop bundles,
Cargo `target/debug`/`target/release` artifacts, and system package prefixes with
lma resource metadata. Use the owning package manager/installer for those.
Program Files is not automatically writable and is not bypassed. Portable or
user-local standalone CLI installations are the intended self-update targets.

The separate **source-build updater** (`sh install/install.sh --update`) retains
its explicit clean-checkout/origin/build contract. It is not used as a fallback
when a release is unavailable. See [source installation](../install/INSTALL.md).

## Validation boundaries

Fixture tests cover versions, selection, platform/backend preservation, checksums,
streamed size/hash failures, binary formats, protected paths, symlinks, locks,
backups, original-file changes and failed/atomic publication. Dry-run CLI tests use
an empty PATH and temporary home. They do not overwrite the running test binary.
A complete real GitHub download/self-replacement cannot be tested until suitable
releases exist; the native release builds/publication workflow still needs CI
validation on all three platforms.
