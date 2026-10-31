'

# Source installation and explicit updates

These installers build **this checkout**, not downloaded executables. They do not
install toolchains, elevate privileges, use `sudo`, or change your PATH. Initial
installation may build local modifications. **Updates require a completely clean
tracked/untracked checkout** and the exact origin
`https://github.com/Hexadecimall/lma.git`.

## Usage

From a complete checkout, including the root and desktop Cargo lockfiles (or the
workspace lockfile if desktop belongs to the workspace):

```sh
# macOS: Metal; Linux: CPU; default two parallel build jobs
sh install/install.sh --dry-run
sh install/install.sh

# Linux / Windows GPU option (requires NVIDIA CUDA toolkit)
sh install/install.sh --cuda --jobs 2
# Linux explicit CPU, without inheriting the root crate's GPU defaults
sh install/install.sh --cpu

# Explicit update: preserve registered backend/jobs unless overridden
sh install/install.sh --update --dry-run
sh install/install.sh --update
```

Windows, from PowerShell:

```powershell
powershell -File install/install.ps1 --dry-run
powershell -File install/install.ps1 --cpu --jobs 2
powershell -File install/install.ps1 --cuda --jobs 2
# Close the desktop app and CLI before updating
powershell -File install/install.ps1 --update
# If Windows refused replacement of a running executable:
powershell -File install/install.ps1 --apply-staged
```

Respect your organization's PowerShell execution policy; no bypass is performed.
`--apply-staged` installs previously built binaries without pulling or rebuilding.
It is Windows-only and cannot be combined with `--update`. Close both executables
before applying, then start the desktop app manually. PATH instructions are printed;
Windows installs to `%LOCALAPPDATA%\lma\bin`. The installer never edits user or
system environment variables.

`--jobs N` accepts 1–64, defaults to 2, and is saved for subsequent updates. Fat LTO
can still require considerable memory even at two jobs; use `--jobs 1` on constrained
systems. macOS desktop always enables Metal, so CPU/CUDA selection is rejected there.

## Prerequisites (install explicitly)

- Python **3.9+**, available as `python3` on Unix or `python`/`python3` on Windows.
- Rust/Cargo **1.89+** (or newer if required by locked dependencies), CMake, and a
  native C/C++ toolchain; libclang must be discoverable by bindgen (`LIBCLANG_PATH`
  if needed). Registry mutations use standard-library cross-process file locking.
- **macOS:** Xcode tools and a working Metal compiler (`xcrun --find metal`). Some
  Xcode versions require the full Xcode installation and its Metal toolchain rather
  than just Command Line Tools. Select the intended Xcode developer directory.
- **Linux:** compiler/build tools, pkg-config, libclang, GTK3, WebKitGTK **4.1**,
  and librsvg development packages. For example, Debian/Ubuntu package names include
  `build-essential cmake pkg-config libclang-dev libgtk-3-dev libwebkit2gtk-4.1-dev
librsvg2-dev`. Other distributions use different names.
- **Windows:** Rust MSVC toolchain, Visual Studio C++ Build Tools, Windows SDK,
  CMake, LLVM/libclang, and the WebView2 runtime. A Developer PowerShell can help
  native build tools find their environment.
- **CUDA:** a compatible NVIDIA driver and CUDA toolkit, including `nvcc` on PATH.
- **Updates only:** `ghx` on PATH. This implementation uses the output format of
  ghx 0.3.3 (`working tree clean` and `origin URL`). Unknown status/remote formats
  are rejected rather than guessed. `ghx pull origin` is documented by that CLI as
  fetching and **fast-forwarding only**; there is no invented `--ff-only` flag.

Cargo may download locked dependencies and execute dependency build scripts. A
source update additionally pulls source from GitHub. No API key or paid service is
required. Your shell/build environment, checkout and dependencies are trusted code;
this is not a sandbox or a signed-release verification system.

## Build, cache, installation and recovery

Both builds use `cargo build --locked --release --jobs N --target-dir SOURCE/target`,
with explicit binary and manifest paths. The root CLI uses
`--no-default-features --features vision,image-gen[,metal|cuda]`. Desktop uses
`--no-default-features --features tauri/custom-protocol` (plus `cuda` when selected);
its macOS dependency enables Metal automatically. `tauri/custom-protocol` ensures
that a plain Cargo build embeds the frontend rather than using a development URL.
The desktop package must embed `../src/runtime/ui` and provide `lma-desktop`.
Neither `cargo-tauri` nor a JavaScript package manager is required by this installer.

Both builds share the project-local Cargo cache. No clean operation is performed.
The root's fat-LTO release profile and desktop's release profile are left unchanged.
An inherited `CARGO_BUILD_TARGET` is removed: these installers install native host
binaries, not cross-compiled artifacts. Custom Cargo config target settings and
other build environment overrides are the caller's responsibility.

| OS      | CLI                              | Desktop                                  |
| ------- | -------------------------------- | ---------------------------------------- |
| macOS   | `~/.local/bin/lma`               | `~/Applications/lma.app`                 |
| Linux   | `~/.local/bin/lma`               | `~/.local/bin/lma-desktop`               |
| Windows | `%LOCALAPPDATA%\lma\bin\lma.exe` | `%LOCALAPPDATA%\lma\bin\lma-desktop.exe` |

Linux also gets `$XDG_DATA_HOME/applications/org.lma.app.desktop` (default
`~/.local/share/applications`). macOS uses `install/Info.plist` for the 0.2.0
`org.lma.app` bundle. If available, `codesign` ad-hoc signs and verifies it before
publication. **This is a local build: not Developer ID signed or notarized.** The
installer does not disable Gatekeeper or remove quarantine attributes.

Only after **both** builds and packaging succeed are all destinations staged beside
their final paths. Files are atomically replaced; the app directory is moved aside
and replaced. Registration is published last. On a handled publication failure,
previous files/directories are restored. There is no single filesystem transaction
covering all destinations: observers can briefly see a mixed version or a missing
app directory, and power loss / forced termination cannot be rolled back reliably.
Rollback failures retain their backup directories and print recovery paths. Build
failures never replace installed files. A successful pull is **not** reverted if a
later build fails; fix the build and rerun using the same cache.

A per-checkout `target/.source-install-lock` excludes concurrent installers, including
external installers racing app-initiated updates. If a process is killed, verify
that no installer/build is still active before removing that lock directory. This
lock does not exclude unrelated manual Cargo builds in the shared target directory;
avoid running those during installation.

Windows permission failures roll back the installed set and stage both new binaries
and registration in `%LOCALAPPDATA%\lma\pending-update`. `--apply-staged` retries
publication. Never terminate a running model process just to replace its executable.

`--dry-run` performs no builds, installs, directory creation, signing or pulls.
With `--update`, it only calls `ghx remote list` and `ghx status` to validate the
checkout, then prints the planned pull/builds. It does not validate every toolchain
prerequisite. An initial dry run does not require build tools.

## Registration and desktop integration contract

The installer publishes `source-install.json` under **`~/.lma/`** on every
platform, matching the application's default configuration directory. On Windows,
`~` is the user's home (`USERPROFILE`). Executables and OS launcher entries retain
the installation locations listed above; external model stores are not moved.
Existing desktop registrations in the former OS-specific app-config directory
remain readable; explicit updates from that directory retain their supplied path.
See [storage policy](../docs/storage.md).

`--config-dir ABSOLUTE_PATH` is available for an explicitly selected application
configuration directory. The updater forwards exactly the directory supplied by
its caller. The JSON schema is:

```json
{
  "schema_version": 1,
  "source_dir": "/absolute/canonical/source/checkout",
  "backend": "metal",
  "jobs": 2,
  "install_locations": {
    "cli": "/Users/example/.local/bin/lma",
    "desktop": "/Users/example/Applications/lma.app"
  }
}
```

Linux additionally records `install_locations.desktop_entry`. Install locations
are informational and always derived from the current user's OS directories;
arbitrary install commands/arguments are not accepted from this JSON. Keep the
checkout and registration private to your user. Symlink registration files, direct
symlink config directories, redirected source paths, symlink installers, and symlink
final install destinations are rejected. This does not defend against another
process with the same user's privileges changing files between validation and use.

`desktop/src/update.rs` only requires `serde` with derive and `serde_json`:

```rust
#[derive(Clone, Default)]
pub struct SourceUpdater;
// Actual implementation shares state across clones.
// pub fn is_running(&self) -> bool
// pub fn status(&self) -> UpdateStatus
// pub fn start(&self, config_dir: PathBuf) -> Result<String, String>
```

`UpdateStatus` serializes `{ state: String, log: String, error: Option<String> }`.
States are **`Idle`, `Running`, `Completed`, `Failed`**, with those exact spellings.
`start` validates the registration synchronously, then launches a dedicated blocking
thread. Keep one managed `SourceUpdater` instance and clone it; independently
constructed instances do not share the in-memory guard (the installer lock still
excludes builds). A second start on the same instance returns an error without
changing the active run. Validation/launch/build failures remain visible in status.

On macOS/Linux it executes `sh SOURCE/install/install.sh --update --backend BACKEND
--jobs N --config-dir CONFIG_DIR` using separate `Command` arguments, never a shell
command string. It captures both output streams concurrently with bounded raw and
displayed logs (last 64 KiB). Invalid UTF-8 bytes are escaped as `\xNN` rather than
discarded. Stream ordering is best-effort, not an exact stdout/stderr timeline.
There is no timeout/cancellation: native release builds can take a long time.

**Only call `start` after explicit user confirmation** that source code will be
pulled, built and executed. The caller must unload the model first and gate model
operations while `is_running()` is true. Never call this automatically on launch
or in the background. Updates stay on the checkout's current branch and use its
origin; no branch switch, stash, reset, unsigned-release claim, quit or relaunch is
performed. `Completed` means installed; the user must restart manually.

Windows returns clear external update/restart instructions without launching a
process. A missing checkout/config or missing GUI process PATH prerequisite is an
explicit error, not an invitation to download/install tools silently. The updater
preserves PATH precedence and appends `~/.cargo/bin`, `~/.local/bin`, `/usr/local/bin`
and (on macOS) `/opt/homebrew/bin`, without sourcing shell profiles. If a desktop
launched from Finder still cannot find tools installed in custom locations, launch
it from a shell with the intended Rust/Python/CMake/ghx environment or run the
external installer in that shell.

## Tests

```sh
python3 -B -m unittest discover -s install -p 'test_*.py' -v
sh -n install/install.sh
# If installed:
shellcheck install/install.sh
# Once the desktop package is present:
cargo test --manifest-path desktop/Cargo.toml update::tests
```

Python tests use temporary homes/checkouts and fake Cargo/ghx/native tools, including
simulated Windows lock failures. They do not compile the app or access a remote.
The Rust module tests use temporary registration and fake shell installers, testing
validation, symlink refusal, log bounds/invalid UTF-8, exclusion and failure/retry.
Platform simulation is not a substitute for real Windows/Linux packaging smoke tests.
