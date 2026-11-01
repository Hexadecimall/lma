# Publishing CLI releases

`.github/workflows/release.yml` builds and publishes **one complete GitHub Release**
from an existing stable tag. It does not build the desktop app, create source
commits, stamp versions, push tags, change the repository/origin, or publish CUDA
assets. Repository operations use Actions checkout and the GitHub API, not shell
Git or GitHub CLI commands. GitHub itself may display its automatic source archives;
they are not the CLI assets described below.

## Current blockers and prerequisites

- **No published releases currently:** the owner's authenticated `ghx api` releases
  query returned `[]` before this pipeline was added. Adding the workflow does not
  itself publish anything; this observation is not a live release-status check.
- **The root `LICENSE` is currently empty and untracked.** The owner must supply
  the correct license text and include it in the tagged commit. Do not invent a
  license or publish until this is resolved. The workflow rejects missing,
  zero-length, or whitespace-only root and llama.cpp licenses before any builds.
- The tag's root `Cargo.toml` must declare package `lma` with exactly the tag's
  version, without the `v`. For example, `v0.3.3` requires `version = "0.3.3"` in
  that commit. Version/lockfile updates are maintainer work, not workflow stamping.
- Commit the release workflow, applicable lockfile/version updates, and notices
  through the normal review process before selecting a release commit. Keep the
  workflow available on the default branch for manual dispatch.
- GitHub Actions must be enabled, and repository policy must permit the publish
  job's `contents: write` token and the selected hosted runners. Private-repository
  runner usage can consume paid Actions minutes. Dependency downloads and native
  prerequisites require network access. No custom publishing secret is required:
  the workflow uses its scoped Actions `GITHUB_TOKEN` through `github-script`.

## Triggers and immutable source selection

1. After reviewing and testing the intended commit, the maintainer creates/pushes
   an appropriate stable tag using their normal process. This workflow does not
   create tags. A pushed tag matching `v*` triggers validation.
2. Alternatively, use **Actions → Publish CLI release → Run workflow** from a
   trusted workflow ref, and provide the required `tag` input for an **existing**
   tag, such as `v0.3.3`. The form's workflow ref does not select the build source:
   checkout always uses the input tag's resolved commit.
3. Validation accepts only `vMAJOR.MINOR.PATCH`, with nonnegative decimal components
   and no leading zeroes except `0`. Branches, missing tags, prereleases, build
   metadata, abbreviated versions, and whitespace are rejected. Lightweight and
   annotated tags are resolved through the API to a commit; tags pointing to other
   object types are rejected.
4. Before any native build, the workflow checks the tagged package version and
   nonempty notices. Every build and the publisher checks out the same resolved
   commit SHA with persisted checkout credentials disabled. The publisher checks
   the tag again and refuses to publish if it has moved. Never force-move a release
   tag; keep release tags immutable (a repository ruleset is recommended).

The workflow operates only on the repository that ran it. There is no hard-coded
owner/repository and no local release/tag operation in this setup procedure.

## Asset contract

For a tag `vX.Y.Z`, the uploaded release assets are exactly:

| Asset                                       | Contents                                                                              |
| ------------------------------------------- | ------------------------------------------------------------------------------------- |
| `lma-vX.Y.Z-aarch64-apple-darwin-metal`     | Raw Apple Silicon CLI executable, Metal enabled                                       |
| `lma-vX.Y.Z-x86_64-unknown-linux-gnu-cpu`   | Raw Linux x86_64 CLI executable, CPU backend                                          |
| `lma-vX.Y.Z-x86_64-pc-windows-msvc-cpu.exe` | Raw Windows x64 MSVC CLI executable, CPU backend                                      |
| `LICENSE`                                   | Exact bytes of the tagged root license                                                |
| `LICENSE-llama.cpp`                         | Exact bytes of `vendor/llama-cpp-sys-2/llama.cpp/LICENSE`, renamed to avoid collision |
| `SHA256SUMS`                                | Checksums for the three binaries and both license notices                             |

These are standalone CLI files, **not** ZIP/tar packages, installers, desktop
bundles, or CUDA builds. Actions artifacts are internal transport only; the final
release receives the raw files extracted from them. No failed CUDA build is
silently replaced with a CPU binary. Additional platforms/backends need their own
explicit asset contract and CI coverage before being added.

`SHA256SUMS` uses standard `sha256sum` text format: a lowercase 64-character SHA-256
hash, two spaces, the exact filename above, and a newline per entry. Filenames do
not include directory prefixes. It covers every payload/notice, not itself. With
all six assets downloaded into one directory, Linux users can check them with
`sha256sum --check SHA256SUMS`; macOS users can use `shasum -a 256 --check SHA256SUMS`.
Raw downloads do not carry Unix executable permission metadata; direct users may
need `chmod +x` on the selected macOS/Linux binary.

Checksums detect corruption and mismatched assets. Since the checksum file is
published in the same release, it is **not an independent cryptographic signature**
or proof against a compromised publishing account. The workflow does not provide
signed checksum manifests, attestations, or reproducible-build guarantees.

## Native build environments

All three matrix entries build only `--bin lma`, using stable Rust and
`cargo build --locked --release --jobs 2 --no-default-features`, an explicit native
`--target`, and the explicit features below. Native CMake parallelism is also
limited to two. There is no desktop build or host-CPU `target-cpu=native` tuning;
implicit ggml AVX/SSE4.2/BMI2/FMA/F16C options are disabled for distribution.

| Runner label   | Required `rustc` host      | Features / prerequisites                                                                          |
| -------------- | -------------------------- | ------------------------------------------------------------------------------------------------- |
| `macos-15`     | `aarch64-apple-darwin`     | `vision,image-gen,speech,metal`; Xcode clang and Metal compiler, Homebrew CMake and LLVM/libclang |
| `ubuntu-24.04` | `x86_64-unknown-linux-gnu` | `vision,image-gen,speech`; build-essential, CMake, Ninja, pkg-config, clang and libclang-dev      |
| `windows-2022` | `x86_64-pc-windows-msvc`   | `vision,image-gen,speech`; x64 MSVC developer environment, CMake, hosted LLVM/libclang            |

GitHub's [hosted-runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
explicitly lists `macos-15` as ARM64 for public and private repositories. This is
an OS-version-pinned Apple Silicon label, not an invented `macos-15-arm64` label
or an ambiguous `macos-latest`. Runner labels/images can change: the workflow
asserts that `rustc --version --verbose` reports exactly the matrix target and
fails rather than emitting a mislabeled binary. It does not cross-compile or
silently substitute an Intel Mac runner.

Linux has an **Ubuntu 24.04 / glibc 2.39 build baseline**, not a manylinux, musl,
fully static, or older-distribution compatibility promise. System C/C++ runtime
requirements also come from that image; older Linux releases require separate
compatibility work and testing. Windows uses MSVC, enables Rust `+crt-static`, and
sets `LLAMA_STATIC_CRT=1` for the native library. llama.cpp is statically linked on
all platforms. System libraries/frameworks are still required; raw standalone
does not mean independent of the operating system.

The updater refuses GNU/Linux replacement on glibc older than 2.39 and macOS
replacement before macOS 15, rather than installing a known incompatible ABI.
The macOS build explicitly sets `MACOSX_DEPLOYMENT_TARGET=15.0`. These minimums
must stay synchronized with `src/update.rs` if release build environments change.

On macOS, the workflow verifies the selected Xcode Metal compiler, downloads the
Metal toolchain component if needed, and requests embedded ggml Metal shaders.
It **ad-hoc signs** the final executable with `codesign --sign -` and verifies that
signature **before** uploading it for aggregate hashing. This is not Developer ID
signing or notarization; Gatekeeper/quarantine handling and distribution trust are
not solved by ad-hoc signing. There is no Apple signing credential in this workflow.

Each matrix job invokes its staged binary with `-v` as a native loading
smoke test. **Real platform build testing in GitHub Actions is still required**
before treating this pipeline or the binaries as production-ready. YAML/static
checks cannot verify native dependency availability, Metal/candle compilation,
MSVC linkage, memory/disk limits, minimum supported OS versions, inference, or
updating/replacing a running executable. The separate cross-platform CI and
platform-specific runtime/updater integration tests must also be reviewed; the
release workflow does not wait on a separate CI workflow or replace those tests.
Stable Rust, major-version action references, runner images, and package-manager
prerequisites may evolve even when Cargo dependencies are locked.

## Draft-first publication and failure recovery

The publish job depends on validation **and the entire successful build matrix**.
It requires exactly the three expected nonempty binary files, copies the two
licenses from the same tagged commit, and computes checksums over the final bytes
(including the macOS ad-hoc signature). Only then does it create one draft release.
It uploads all six assets, verifies the API asset names, uploaded state, sizes,
and server SHA-256 digests for payloads/notices when supplied by GitHub, and only
then changes the draft to a published stable/latest release.

Same-tag push/dispatch runs are serialized without cancelling an upload in
progress. Both validation and publishing reject an existing release for the tag;
the write-authorized publisher also checks drafts. Nothing overwrites assets,
edits an existing release, or deletes a failed draft automatically.

- If validation or a build fails before a draft exists, fix the underlying issue.
  Retry the same unchanged tag only if its source is correct; source changes need
  a new reviewed commit/tag, not version stamping or a moved release tag.
- If creation/upload/verification/publication fails, any created draft stays
  unpublished for owner inspection. A rerun will refuse that existing draft. The
  owner must explicitly resolve/remove an incomplete draft in GitHub before
  retrying, or choose a new release tag. Do not manually publish a partial draft.
- If the final API response is lost, inspect GitHub first: publication may already
  have succeeded. A rerun must not overwrite a completed release.

## Private-repository updater access

For a private repository, users need a `GH_TOKEN` or `GITHUB_TOKEN` in the updater's
environment with access to that repository and read access to its contents/release
assets (and organization SSO authorization if required). Anonymous requests cannot
list/download private releases; GitHub may return 404 rather than reveal their
existence. A local GitHub CLI login alone does not necessarily put a token in the
native updater's environment. Configure credentials through the user's secure
secret-management mechanism, never in committed files, command examples containing
real tokens, release notes, or binaries. The workflow's automatic token exists only
inside Actions and is not distributed to CLI users.
