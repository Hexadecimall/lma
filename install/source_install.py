#!/usr/bin/env python3
"""User-local source builds. No network action other than explicit Cargo/ghx work."""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ORIGIN = "https://github.com/Hexadecimall/lma.git"
CONFIG_NAME = "source-install.json"


def run(argv, cwd, capture=False):
    print("+ " + subprocess.list2cmdline([str(a) for a in argv]), flush=True)
    env = dict(os.environ, NO_COLOR="1", CLICOLOR="0", CARGO_TERM_COLOR="never")
    # Always use a single project-local cache, independent of caller environment.
    env.pop("CARGO_BUILD_TARGET", None)
    result = subprocess.run(
        [str(a) for a in argv],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if result.returncode:
        detail = (result.stderr or b"").decode("utf-8", errors="backslashreplace")
        raise RuntimeError(f"{argv[0]} failed ({result.returncode}). {detail}")
    return (result.stdout or b"").decode("utf-8", errors="strict")


def verify_update(source):
    # Fail closed on unknown output formats; never infer cleanliness from an exit code.
    remotes = run(["ghx", "remote", "list"], source, capture=True)
    origins = [
        line.split()
        for line in remotes.splitlines()
        if line.split() and line.split()[0] == "origin"
    ]
    if origins != [["origin", ORIGIN]]:
        raise RuntimeError(f"Refusing update: origin must be exactly {ORIGIN}")
    status = run(["ghx", "status"], source, capture=True).strip()
    if status != "working tree clean":
        raise RuntimeError(
            "Refusing update: checkout is dirty or ghx status is unrecognized. "
            "Commit or move ALL tracked/untracked changes yourself.\n" + status
        )


def locations(system, home, desktop=False):
    if system == "Darwin":
        paths = {"cli": home / ".local/bin/lma"}
        if desktop:
            paths["desktop"] = home / "Applications/lma.app"
        return home / ".lma", paths
    if system == "Linux":
        paths = {"cli": home / ".local/bin/lma"}
        if desktop:
            data = Path(os.environ.get("XDG_DATA_HOME") or home / ".local/share")
            if not data.is_absolute():
                raise RuntimeError("XDG_DATA_HOME must be an absolute path")
            paths.update({
                "desktop": home / ".local/bin/lma-desktop",
                "desktop_entry": data / "applications/org.lma.app.desktop",
            })
        return home / ".lma", paths
    if system == "Windows":
        if not os.environ.get("LOCALAPPDATA"):
            raise RuntimeError("LOCALAPPDATA must be set")
        root = Path(os.environ["LOCALAPPDATA"]) / "lma"
        paths = {"cli": root / "bin/lma.exe"}
        if desktop:
            paths["desktop"] = root / "bin/lma-desktop.exe"
        return home / ".lma", paths
    raise RuntimeError(f"Unsupported OS: {system}")


def require_tools(system, backend, update, desktop=False):
    tools = {
        "cargo": "Install stable Rust with rustup explicitly (rustup.rs).",
        "rustc": "Install stable Rust with rustup explicitly (rustup.rs).",
        "cmake": "Install CMake and ensure it is on PATH.",
    }
    if update:
        tools["ghx"] = "Install ghx; updates require ghx remote list/status/pull."
    if system == "Darwin":
        tools["xcrun"] = "Install Xcode Command Line Tools: xcode-select --install."
    elif system == "Linux":
        tools["cc"] = "Install a C/C++ build toolchain and libclang development packages."
        if desktop:
            tools["pkg-config"] = "Install pkg-config, WebKitGTK 4.1, GTK3 and librsvg development packages."
    if backend == "cuda":
        tools["nvcc"] = (
            "Install a compatible NVIDIA CUDA toolkit and driver; put nvcc on PATH."
        )
    for tool, hint in tools.items():
        if not shutil.which(tool):
            raise RuntimeError(f"Missing prerequisite: {tool}. {hint}")
    if system == "Darwin":
        try:
            run(["xcrun", "--find", "clang"], Path.cwd(), capture=True)
            if backend == "metal" or desktop:
                run(["xcrun", "--find", "metal"], Path.cwd(), capture=True)
        except RuntimeError as error:
            raise RuntimeError(
                "Xcode C/Metal tools unavailable. Install/select Xcode and its Metal toolchain; "
                "Command Line Tools alone may not include Metal. " + str(error)
            ) from error
    if system == "Linux" and desktop:
        try:
            run(
                ["pkg-config", "--exists", "webkit2gtk-4.1", "gtk+-3.0", "librsvg-2.0"],
                Path.cwd(),
                capture=True,
            )
        except RuntimeError as error:
            raise RuntimeError(
                "Missing Linux desktop development packages: install WebKitGTK 4.1, GTK3, "
                "and librsvg development packages. See install/INSTALL.md. "
                + str(error)
            ) from error
    if system == "Windows":
        print(
            "Requires Visual Studio C++ Build Tools, Windows SDK and LLVM/libclang. "
            + ("Desktop also requires WebView2. " if desktop else "")
            + "Use a Developer PowerShell if your tools are not on PATH."
        )


def build_commands(source, backend, jobs, desktop=False):
    common = [
        "--locked",
        "--release",
        "--jobs",
        str(jobs),
        "--target-dir",
        str(source / "target"),
    ]
    # Explicitly retain functional features without inheriting the root GPU default.
    core_features = "vision,image-gen,speech" + (
        "," + backend if backend != "cpu" else ""
    )
    cli = [
        "cargo",
        "build",
        "--manifest-path",
        str(source / "Cargo.toml"),
        "--bin",
        "lma",
        "--no-default-features",
        "--features",
        core_features,
    ] + common
    if not desktop:
        return [cli]
    desktop = [
        "cargo",
        "build",
        "--manifest-path",
        str(source / "desktop/Cargo.toml"),
        "--bin",
        "lma-desktop",
        "--no-default-features",
        "--features",
        "tauri/custom-protocol",
    ] + common
    if backend == "cuda":
        desktop += ["--features", "cuda"]
    return [cli, desktop]


def remove(path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def publish(items):
    """Stage beside each destination; rename atomically per item; rollback the set."""
    stages = []
    try:
        for src, dest in items:
            if dest.is_symlink():
                raise RuntimeError(f"Refusing symlink install destination: {dest}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix=".lma-install-", dir=dest.parent))
            entry = {"stage": stage, "dest": dest, "backed": False, "installed": False}
            stages.append(entry)
            if src.is_dir():
                shutil.copytree(src, stage / "new")
            else:
                shutil.copy2(src, stage / "new")
        for entry in stages:
            dest, stage = entry["dest"], entry["stage"]
            if dest.exists():
                if dest.is_dir():
                    os.replace(dest, stage / "backup")
                else:
                    # Keep a file at its public path until atomic replacement.
                    shutil.copy2(dest, stage / "backup")
                entry["backed"] = True
            os.replace(stage / "new", dest)
            entry["installed"] = True
    except BaseException:
        for entry in reversed(stages):
            dest, stage = entry["dest"], entry["stage"]
            try:
                if entry["installed"]:
                    remove(dest)
                if entry["backed"]:
                    # A failed Windows replacement leaves the old executable intact.
                    if not entry["installed"] and dest.is_file():
                        continue
                    os.replace(stage / "backup", dest)
            except OSError as error:
                print(
                    f"ROLLBACK NEEDS HELP: {error}. Backup retained in {stage}",
                    file=sys.stderr,
                )
                entry["retain"] = True
        raise
    finally:
        for entry in stages:
            if not entry.get("retain"):
                shutil.rmtree(entry["stage"], ignore_errors=True)


def desktop_entry(binary):
    # Desktop Entry Exec has its own quoting rules (not shell quoting).
    value = str(binary)
    if any(c in value for c in "\n\r\x00"):
        raise RuntimeError("Desktop entry paths cannot contain newlines or NUL")
    value = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("`", "\\`")
        .replace("$", "\\$")
        .replace("%", "%%")
    )
    # String-value escaping is a separate layer from Exec escaping.
    value = value.replace("\\", "\\\\")
    return (
        '[Desktop Entry]\nType=Application\nName=lma\nExec="'
        + value
        + '"\nTerminal=false\nCategories=Utility;\n'
    )


def package_and_install(source, system, config_dir, paths, config, apply_staged=False):
    release = source / "target/release"
    pending = paths["cli"].parent.parent / "pending-update"
    if apply_staged:
        release = pending
    with tempfile.TemporaryDirectory(prefix="lma-package-") as temporary:
        package = Path(temporary)
        suffix = ".exe" if system == "Windows" else ""
        cli, desktop = release / ("lma" + suffix), release / ("lma-desktop" + suffix)
        binaries = [cli, desktop] if config.get("desktop", True) else [cli]
        for binary in binaries:
            if not binary.is_file() or binary.is_symlink():
                raise RuntimeError(f"Build output missing or symlinked: {binary}")
        items = [(cli, paths["cli"])]
        if config.get("desktop", True) and system == "Darwin":
            app = package / "lma.app"
            (app / "Contents/MacOS").mkdir(parents=True)
            shutil.copy2(desktop, app / "Contents/MacOS/lma-desktop")
            shutil.copy2(source / "install/Info.plist", app / "Contents/Info.plist")
            if shutil.which("codesign"):
                run(["codesign", "--force", "--sign", "-", app], source)
                run(["codesign", "--verify", "--deep", "--strict", app], source)
            else:
                print(
                    "WARNING: codesign unavailable; bundle is unsigned. This is a local, non-notarized build."
                )
            items.append((app, paths["desktop"]))
        elif config.get("desktop", True):
            items.append((desktop, paths["desktop"]))
        if config.get("desktop", True) and system == "Linux":
            entry = package / "org.lma.app.desktop"
            entry.write_text(desktop_entry(paths["desktop"]), encoding="utf-8")
            items.append((entry, paths["desktop_entry"]))
        registration = package / CONFIG_NAME
        registration.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        # Registration is committed last, and rolled back along with executables.
        items.append((registration, config_dir / CONFIG_NAME))
        try:
            publish(items)
        except PermissionError as error:
            if system != "Windows" or apply_staged:
                raise
            pending.mkdir(parents=True, exist_ok=True)
            publish(
                [
                    *((binary, pending / binary.name) for binary in binaries),
                    (registration, pending / CONFIG_NAME),
                ]
            )
            raise RuntimeError(
                f"Installation was rolled back: {error}. New binaries staged in {pending}. "
                + ("Close lma and lma-desktop" if config.get("desktop", True) else "Close lma")
                + ", then run install/install.ps1 --apply-staged from this checkout "
                "without overrides. Restart manually."
            ) from error
    if system == "Windows" and pending.exists():
        shutil.rmtree(pending)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build/install this checkout locally. Never installs toolchains."
    )
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--cpu", dest="backend", action="store_const", const="cpu")
    backend.add_argument("--cuda", dest="backend", action="store_const", const="cuda")
    backend.add_argument("--backend", choices=["cpu", "metal", "cuda"])
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument(
        "--desktop", action="store_true",
        help="also build/install desktop (default: CLI only, including updates)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="verify origin/clean checkout, then ghx pull origin (fast-forward only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print plan; do not build, write, install or pull",
    )
    parser.add_argument(
        "--apply-staged",
        action="store_true",
        help="Windows only: install previously staged binaries after closing app",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        help="registration directory (default: ~/.lma)",
    )
    args = parser.parse_args(argv)
    source = Path(__file__).resolve().parent.parent
    system, home = platform.system(), Path.home()
    config_dir, paths = locations(system, home)
    if args.config_dir:
        config_dir = args.config_dir
    if not config_dir.is_absolute() or config_dir.is_symlink():
        raise RuntimeError("Config directory must be absolute and not a symlink")
    if args.apply_staged and (system != "Windows" or args.update):
        raise RuntimeError(
            "--apply-staged is Windows-only and cannot be combined with --update"
        )
    if args.apply_staged and (
        args.backend is not None or args.jobs is not None or args.desktop
    ):
        raise RuntimeError(
            "--apply-staged uses staged components/backend/jobs; do not override them"
        )
    saved = None
    saved_path = config_dir / CONFIG_NAME
    if args.apply_staged:
        saved_path = paths["cli"].parent.parent / "pending-update" / CONFIG_NAME
    if (args.update or args.apply_staged) and saved_path.exists():
        if saved_path.is_symlink() or not saved_path.is_file():
            raise RuntimeError("Registration must be a regular non-symlink file")
        with saved_path.open("rb") as handle:
            data = handle.read(65537)
        if len(data) > 65536:
            raise RuntimeError("Source registration exceeds 64 KiB")
        saved = json.loads(data)
        if (
            not isinstance(saved, dict)
            or saved.get("schema_version") != 1
            or saved.get("source_dir") != str(source)
        ):
            raise RuntimeError(
                "Registration does not belong to this source checkout/schema"
            )
        if "desktop" in saved and type(saved["desktop"]) is not bool:
            raise RuntimeError("Registration desktop must be a boolean")
    if args.apply_staged and saved is None:
        raise RuntimeError("No staged update found")
    # Normal runs never inherit desktop opt-in; staged runs must preserve their set.
    desktop = saved.get("desktop", True) if args.apply_staged else args.desktop
    _, paths = locations(system, home, desktop)
    selected = (
        args.backend
        or (saved or {}).get("backend")
        or ("metal" if system == "Darwin" else "cpu")
    )
    jobs = args.jobs if args.jobs is not None else (saved or {}).get("jobs", 2)
    if type(jobs) is not int or not 1 <= jobs <= 64:
        raise RuntimeError("--jobs must be between 1 and 64 (default 2)")
    if selected not in ("metal", "cpu", "cuda"):
        raise RuntimeError("Invalid backend")
    if (system == "Darwin" and desktop and selected != "metal") or (
        system != "Darwin" and selected == "metal"
    ):
        raise RuntimeError(
            "macOS desktop always enables Metal; use --backend metal with --desktop. "
            "Metal is supported only on macOS."
        )
    manifests = [source / "Cargo.toml", source / "Cargo.lock"]
    if desktop:
        manifests.append(source / "desktop/Cargo.toml")
    for file in [] if args.apply_staged else manifests:
        if not file.is_file():
            raise RuntimeError(
                f"Missing {file}; use a complete source checkout with committed lockfiles."
            )
    config = {
        "schema_version": 1,
        "source_dir": str(source),
        "backend": selected,
        "jobs": jobs,
        "desktop": desktop,
        "install_locations": {key: str(value) for key, value in paths.items()},
    }
    commands = [] if args.apply_staged else build_commands(source, selected, jobs, desktop)
    print(
        f"Source: {source}\nBackend: {selected}; jobs: {jobs}; shared cache: {source / 'target'}"
    )
    for key, value in paths.items():
        print(f"Install {key}: {value}")
    print(f"Registration: {config_dir / CONFIG_NAME}")
    if args.dry_run:
        if args.update:
            verify_update(source)
            print("Would run: ghx pull origin (current branch, fast-forward only)")
        for command in [] if args.apply_staged else commands:
            print("Would run: " + subprocess.list2cmdline(command))
        print(
            "Would stage, publish and register only after successful builds; no changes made."
        )
        return
    if not args.apply_staged:
        require_tools(system, selected, args.update, desktop)
    target = source / "target"
    target.mkdir(exist_ok=True)
    lock = target / ".source-install-lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise RuntimeError(
            f"Another installer is active. If it crashed, verify no build is running before removing {lock}."
        ) from error
    try:
        if args.update:
            verify_update(source)
            run(["ghx", "pull", "origin"], source)
        if not args.apply_staged:
            for command in commands:
                run(command, source)
        package_and_install(
            source, system, config_dir, paths, config, args.apply_staged
        )
    finally:
        lock.rmdir()
    print("Completed. " + (
        "Restart the desktop app manually to use the new build."
        if desktop else "CLI installed; existing desktop installations were left unchanged."
    ))
    print(
        f"CLI PATH: add {paths['cli'].parent} to your user PATH if it is not already present."
    )
    if system == "Darwin" and desktop:
        print(
            "Local source build: ad-hoc signed if codesign is available; NOT Developer ID signed or notarized."
        )


if __name__ == "__main__":
    try:
        if sys.version_info < (3, 9):
            raise RuntimeError(
                "Python 3.9+ is required; install a supported Python explicitly"
            )
        main()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"Install failed: {error}", file=sys.stderr)
        sys.exit(1)
