# Source installation and explicit updates

These installers build **this checkout**, not downloaded executables. They do not
install toolchains, elevate privileges, use `sudo`, or change your PATH. Initial
installation may build local modifications. **Updates require a completely clean
tracked/untracked checkout** and the exact origin
`https://github.com/Hexadecimall/lma.git`.

## Usage

From a checkout with the root Cargo manifest and lockfile. **Normal runs and
`--update` build/install only the CLI.** Add `--desktop` on each invocation to
build/install both CLI and desktop; saved registrations never opt normal runs in.
CLI-only runs do not remove or replace existing GUI binaries, bundles or launchers
and do not require the desktop checkout, Tauri, or webview dependencies. Desktop is
optional; this installer does not implement menu-bar or other new GUI features.

```sh
# CLI only: macOS Metal; Linux CPU; default two parallel build jobs
sh install/install.sh --dry-run
sh install/install.sh

# Linux / Windows GPU option (requires NVIDIA CUDA toolkit)
sh install/install.sh --cuda --jobs 2
# Explicit CLI CPU: omit explicit GPU features (see macOS caveat below)
sh install/install.sh --cpu

# Opt into CLI + desktop, requiring platform webview dependencies
sh install/install.sh --desktop

# Explicit CLI-only update: preserve registered backend/jobs unless overridden
sh install/install.sh --update --dry-run
sh install/install.sh --update
# Update both components explicitly (macOS desktop requires Metal)
sh install/install.sh --update --desktop
```

Windows, from PowerShell:

```powershell
powershell -File install/install.ps1 --dry-run
powershell -File install/install.ps1 --cpu --jobs 2
powershell -File install/install.ps1 --cuda --jobs 2
# CLI-only update; close the CLI first
powershell -File install/install.ps1 --update
# Opt into both components; close the desktop app and CLI first
powershell -File install/install.ps1 --update --desktop
# If Windows refused replacement of a running executable:
powershell -File install/install.ps1 --apply-staged
```

Respect your organization's PowerShell execution policy; no bypass is performed.
`--apply-staged` installs previously built binaries without pulling or rebuilding.
It is Windows-only and cannot be combined with `--update`, `--desktop`, backend
flags or `--jobs`. It follows the staged component selection, not current defaults:
CLI-only staging neither requires nor replaces a desktop binary; legacy staging
without a `desktop` field still requires both binaries. Close the affected
executables before applying, then restart manually. PATH instructions are printed;
Windows installs to `%LOCALAPPDATA%\lma\bin`. The installer never edits user or
system environment variables.

`--jobs N` accepts 1–64, defaults to 2, and is saved for subsequent updates. Fat LTO
can still require considerable memory even at two jobs; use `--jobs 1` on constrained
systems. macOS defaults to Metal. CLI-only `--cpu` omits explicit GPU features,
but llama dependencies may still enable implicit Metal on Apple Silicon; use
runtime `lma s NAME -g cpu` to disable offload. With `--desktop`, macOS always
enables Metal and rejects CPU/CUDA selection; use `--backend metal` if a saved
CLI backend was CPU. Non-macOS Metal selection is invalid.

## System-prefix package staging

Normal source installations remain user-local and CLI-only by default. To prepare
files for a system package **without installing them into protected directories**,
use `--package-dir ABSOLUTE_PATH`. The output directory must not already exist;
its parent must exist. `--prefix ABSOLUTE_PATH` overrides the final prefix and is
valid only in package mode. These are installer flags, not root lma CLI flags.

| Host    | Default final prefix                                     | Staged path beneath PACKAGE_DIR |
| ------- | -------------------------------------------------------- | ------------------------------- |
| Windows | `%ProgramFiles%\\lma` (usually `C:\\Program Files\\lma`) | `lma/bin/lma.exe`               |
| Linux   | `/usr`                                                   | `usr/bin/lma`                   |
| macOS   | `/usr/local`                                             | `usr/local/bin/lma`             |
| FreeBSD | `/usr/local`                                             | `usr/local/bin/lma`             |

Unix package roots mirror `/`. Windows package roots represent the final prefix's
parent: a final `D:\\Apps\\lma` also stages as `lma/...`; the manifest records the
full intended prefix. There is no cross-compilation or platform simulation flag.
FreeBSD is **experimental CPU CLI only**; desktop, Metal and CUDA are rejected.
User-local FreeBSD CLI installation uses `~/.local/bin/lma`.

```sh
# Choose a new output directory outside the source checkout; preview writes nothing.
sh install/install.sh --package-dir /absolute/package-output --dry-run
# Build and atomically publish that new package tree, after supplying a real license.
sh install/install.sh --package-dir /absolute/package-output
# Override final prefix; optional desktop remains a separate opt-in.
sh install/install.sh --package-dir /absolute/another-output --prefix /opt/lma --desktop
```

PowerShell accepts the same arguments through `install/install.ps1`. No automatic
elevation, system installation, PATH editing or package-manager registration is
performed. Distribution packaging/installers must own privileged publication and
runtime dependencies; do not run source builds as administrator/root merely to
write a staging tree. This is **not** an MSI, deb/rpm, FreeBSD port or notarized
macOS installer.

Each final prefix contains:

```text
bin/lma[.exe]
bin/lma-desktop[.exe]               # only with --desktop; plain binary, not .app
share/lma/appicon.png
share/lma/icon.ico                 # when present in the source inventory
share/lma/package-manifest.json
share/licenses/lma/LICENSE
share/licenses/lma/llama.cpp/LICENSE
```

The manifest records version, host, backend, selected components, final prefix and
file inventory. The currently inventoried images are PNG/ICO icons; no other named
image directory is manufactured. The macOS release was inspected with `otool -L`:
llama.cpp is statically linked and only system libraries/frameworks are required.
No cosmetic `lib/llama.cpp` directory or arbitrary build-output libraries are
copied. Future private dynamic libraries must be explicitly inventoried with
loader paths and notices; CUDA/desktop host dependencies are not bundled here.

Both license inputs and required assets must be nonempty regular non-symlink
files. **The current root `LICENSE` is empty: actual package publication is blocked
until the project owner supplies its license.** The installer never invents or
overwrites it. Dry runs still work without assets, licenses or built binaries.
This inventory is not a complete third-party license-compliance audit.

Package mode never creates `~/.lma/source-install.json` or touches a local install.
It cannot combine with `--update`, `--apply-staged` or `--config-dir`; system
packages must be updated by their package manager, not the user-local source
updater. Models/settings/caches remain per-user under `~/.lma`, not the system
prefix. Traversal, symlink/reparse-point paths, existing outputs and protected
source/state overlaps are rejected. Failed builds/copies/publication leave no
completed package; the normal project `target/` build cache may still change.

## Prerequisites (install explicitly)

- Python **3.9+**, available as `python3` on Unix or `python`/`python3` on Windows.
- Rust/Cargo **1.89+** (or newer if required by locked dependencies), CMake, and a
  native C/C++ toolchain; libclang must be discoverable by bindgen (`LIBCLANG_PATH`
  if needed). Registry mutations use standard-library cross-process file locking.
- **macOS:** Xcode tools; Metal builds (the default, and required with `--desktop`)
  also require a working Metal compiler (`xcrun --find metal`). Some
  Xcode versions require the full Xcode installation and its Metal toolchain rather
  than just Command Line Tools. Select the intended Xcode developer directory.
- **Linux CLI:** compiler/build tools, CMake and libclang (for example,
  `build-essential cmake libclang-dev` on Debian/Ubuntu). **Only with `--desktop`:**
  pkg-config, GTK3, WebKitGTK **4.1** and librsvg development packages (for example,
  `pkg-config libgtk-3-dev libwebkit2gtk-4.1-dev librsvg2-dev`). CLI-only runs do not
  check pkg-config or webview packages. Other distributions use different names.
- **Windows:** Rust MSVC toolchain, Visual Studio C++ Build Tools, Windows SDK,
  CMake and LLVM/libclang; **only with `--desktop`**, the WebView2 runtime.
  A Developer PowerShell can help
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

The default performs **one CLI build**. `--desktop` adds one desktop build.
Selected builds use `cargo build --locked --release --jobs N --target-dir SOURCE/target`,
with explicit binary and manifest paths. The root CLI uses
`--no-default-features --features vision,image-gen,speech[,metal|cuda]`. Desktop uses
`--no-default-features --features tauri/custom-protocol` (plus `cuda` when selected);
its macOS dependency enables Metal automatically. `tauri/custom-protocol` ensures
that a plain Cargo build embeds the frontend rather than using a development URL.
With `--desktop`, the desktop package must embed `../src/runtime/ui` and provide
`lma-desktop`; its manifest is not checked for CLI-only runs.
Neither `cargo-tauri` nor a JavaScript package manager is required by this installer.

All selected builds share the project-local Cargo cache. No clean operation is performed.
The root's fat-LTO release profile and desktop's release profile are left unchanged.
An inherited `CARGO_BUILD_TARGET` is removed: these installers install native host
binaries, not cross-compiled artifacts. Custom Cargo config target settings and
other build environment overrides are the caller's responsibility.

| OS      | CLI (default)                    | Desktop (`--desktop` only)               |
| ------- | -------------------------------- | ---------------------------------------- |
| macOS   | `~/.local/bin/lma`               | `~/Applications/lma.app`                 |
| Linux   | `~/.local/bin/lma`               | `~/.local/bin/lma-desktop`               |
| Windows | `%LOCALAPPDATA%\lma\bin\lma.exe` | `%LOCALAPPDATA%\lma\bin\lma-desktop.exe` |

With `--desktop`, Linux also gets
`$XDG_DATA_HOME/applications/org.lma.app.desktop` (default
`~/.local/share/applications`). With `--desktop`, macOS uses `install/Info.plist` for the 0.3.4
`org.lma.app` bundle. If available, `codesign` ad-hoc signs and verifies it before
publication. **This is a local build: not Developer ID signed or notarized.** The
installer does not disable Gatekeeper or remove quarantine attributes. CLI-only
macOS installation skips app packaging and codesign entirely.

Only after **all selected** builds and packaging succeed are their destinations staged beside
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

Windows permission failures roll back the installed set and stage only the selected
new binaries and registration in `%LOCALAPPDATA%\lma\pending-update`. `--apply-staged` retries
publication using that staged component set, without backend/jobs/component
overrides. Never terminate a running model process just to replace its executable.

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
  "desktop": false,
  "install_locations": {
    "cli": "/Users/example/.local/bin/lma"
  }
}
```

New registrations include an exact boolean `desktop` and only selected install
locations: CLI-only has just `cli`; `--desktop` records `desktop: true`, adds
`install_locations.desktop`, and on Linux adds `install_locations.desktop_entry`.
The optional field keeps schema version 1: legacy metadata with no `desktop` field
means both components for validation and staged application. A present non-boolean
(including null or 0/1) is invalid. Normal install/update runs still require explicit
`--desktop`, even with legacy or saved `desktop: true` metadata. CLI locations must
always be absolute; desktop locations must be present and absolute for legacy or
`desktop: true` metadata. Install locations
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

On macOS/Linux it executes `sh SOURCE/install/install.sh --update --desktop --backend BACKEND
--jobs N --config-dir CONFIG_DIR` using separate `Command` arguments, never a shell
command string. It captures both output streams concurrently with bounded raw and
displayed logs (last 64 KiB). Invalid UTF-8 bytes are escaped as `\xNN` rather than
discarded. Stream ordering is best-effort, not an exact stdout/stderr timeline.
There is no timeout/cancellation: native release builds can take a long time.
A GUI-initiated update always explicitly includes desktop, even if the current
registration is CLI-only. On macOS it selects Metal for the GUI, including when
CLI-only metadata previously selected CPU.

**Only call `start` after explicit user confirmation** that source code will be
pulled, built and executed. The caller must unload the model first and gate model
operations while `is_running()` is true. Never call this automatically on launch
or in the background. Updates stay on the checkout's current branch and use its
origin; no branch switch, stash, reset, unsigned-release claim, quit or relaunch is
performed. `Completed` means installed; the user must restart manually.

Windows returns external `install/install.ps1 --update --desktop` instructions
without launching a process. For a staged update use `--apply-staged` alone, with
no component/backend/jobs overrides. A missing checkout/config or missing GUI process PATH prerequisite is an
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
# Headless desktop-state/updater tests; no native desktop/webview build:
cargo test --test desktop_state
```

Python tests use temporary homes/checkouts and fake Cargo/ghx/native tools, including
simulated Windows lock failures. They do not compile the app or access a remote.
The Rust module tests use temporary registration and fake shell installers, testing
validation, symlink refusal, log bounds/invalid UTF-8, exclusion and failure/retry.
Platform simulation is not a substitute for real Windows/Linux packaging smoke tests.
