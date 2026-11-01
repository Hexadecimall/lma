#!/usr/bin/env python3
"""User-local installs or isolated package roots built for the current host."""

import argparse
import json
import os
import platform
import re
import shutil
import stat
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
    if system in ("Linux", "FreeBSD"):
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
    elif system in ("Linux", "FreeBSD"):
        tools["cc"] = "Install a C/C++ build toolchain and libclang development packages."
        if desktop and system == "Linux":
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


def validate_selection(system, selected, jobs, desktop):
    if system not in ("Linux", "Darwin", "FreeBSD", "Windows"):
        raise RuntimeError(f"Unsupported OS: {system}")
    if type(jobs) is not int or not 1 <= jobs <= 64:
        raise RuntimeError("--jobs must be between 1 and 64 (default 2)")
    if selected not in ("metal", "cpu", "cuda"):
        raise RuntimeError("Invalid backend")
    if system == "FreeBSD" and (desktop or selected != "cpu"):
        raise RuntimeError("FreeBSD is experimental: CPU CLI only; no desktop or CUDA")
    if (system == "Darwin" and desktop and selected != "metal") or (
        system != "Darwin" and selected == "metal"
    ):
        raise RuntimeError(
            "macOS desktop always enables Metal; use --backend metal with --desktop. "
            "Metal is supported only on macOS."
        )
    if system == "FreeBSD":
        print("WARNING: FreeBSD CPU CLI support is experimental.")


def checked_absolute_path(path, label):
    # Check before resolve(): resolution would conceal traversal and symlink parents.
    if not path.is_absolute() or ".." in path.parts or any(
        c in str(path) for c in "\x00\n\r"
    ):
        raise RuntimeError(f"{label} must be absolute and contain no traversal or control characters")
    if path == Path(path.anchor):
        raise RuntimeError(f"{label} must not be a filesystem root")
    for part in (path, *path.parents):
        if part.is_symlink():
            raise RuntimeError(f"{label} must not contain symlinks: {part}")
        try:
            attributes = part.lstat()
        except FileNotFoundError:
            continue
        except NotADirectoryError as error:
            raise RuntimeError(f"{label} has a non-directory ancestor: {part}") from error
        # Junctions and other Windows reparse points can redirect writes too.
        if getattr(attributes, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise RuntimeError(f"{label} must not contain reparse points: {part}")
    return path


def check_package_destination(destination, source, home):
    checked_absolute_path(destination, "--package-dir")
    for protected in (source.resolve(), home.resolve() / ".lma"):
        overlaps = (destination == protected or protected in destination.parents
                    or destination in protected.parents)
        # Path comparisons alone miss case aliases on case-insensitive POSIX volumes.
        if protected.exists():
            overlaps = overlaps or any(
                parent.exists() and parent.samefile(protected)
                for parent in destination.parents
            )
        if overlaps:
            raise RuntimeError(f"--package-dir overlaps protected path: {protected}")
    if destination == home.resolve() or destination in home.resolve().parents:
        raise RuntimeError("--package-dir must not be the home directory or its ancestor")
    if destination.exists():
        raise RuntimeError(f"--package-dir must not exist: {destination}")
    if not destination.parent.is_dir():
        raise RuntimeError("--package-dir parent must be an existing directory")


def required_package_file(path):
    checked_absolute_path(path, "Package input")
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Package input must be a nonempty regular non-symlink file: {path}")


def package_version(source):
    # Python 3.9 has no tomllib; read only the literal [package] version, not dependencies.
    text = (source / "Cargo.toml").read_text(encoding="utf-8")
    package = re.search(r"(?ms)^\[package\]\s*\n(.*?)(?=^\[|\Z)", text)
    version = re.search(r'^version\s*=\s*"([^"\n]+)"\s*(?:#.*)?$',
                        package.group(1), re.MULTILINE) if package else None
    if not version:
        raise RuntimeError("Cargo.toml must contain a literal [package] version for packaging")
    return version.group(1)


def stage_package(args, source, system, home):
    """Build normally, but publish only a new package tree; never register or elevate."""
    if args.update or args.apply_staged or args.config_dir is not None:
        raise RuntimeError("--package-dir cannot be combined with --update, --apply-staged or --config-dir")
    selected = args.backend or ("metal" if system == "Darwin" else "cpu")
    jobs = args.jobs if args.jobs is not None else 2
    validate_selection(system, selected, jobs, args.desktop)
    destination = args.package_dir
    check_package_destination(destination, source, home)
    if args.prefix is not None:
        prefix = args.prefix
    elif system == "Windows":
        if not os.environ.get("ProgramFiles"):
            raise RuntimeError("ProgramFiles must be set for the default Windows package prefix")
        program_files = Path(os.environ["ProgramFiles"])
        checked_absolute_path(program_files, "ProgramFiles")
        prefix = program_files / "lma"
    else:
        prefix = Path("/usr" if system == "Linux" else "/usr/local")
    checked_absolute_path(prefix, "--prefix")
    for part in (prefix, *prefix.parents):
        if part.exists() and not part.is_dir():
            raise RuntimeError(f"--prefix must name a directory, not an existing file: {part}")
    # Windows package root represents prefix.parent, not a drive root or Program Files.
    # Thus C:\\Program Files\\lma and D:\\Apps\\lma both stage as lma/bin/lma.exe.
    root = Path(prefix.name) if system == "Windows" else prefix.relative_to(prefix.anchor)
    suffix = ".exe" if system == "Windows" else ""
    binaries = [(source / "target/release" / ("lma" + suffix), root / "bin" / ("lma" + suffix))]
    if args.desktop:
        binaries.append((source / "target/release" / ("lma-desktop" + suffix),
                         root / "bin" / ("lma-desktop" + suffix)))
    assets = [
        (source / "desktop/icons/icon.png", root / "share/lma/appicon.png"),
        (source / "LICENSE", root / "share/licenses/lma/LICENSE"),
        (source / "vendor/llama-cpp-sys-2/llama.cpp/LICENSE",
         root / "share/licenses/lma/llama.cpp/LICENSE"),
    ]
    # Preserve the other inventoried image by name; never glob arbitrary build assets.
    icon = source / "desktop/icons/icon.ico"
    if icon.exists() or icon.is_symlink():
        assets.append((icon, root / "share/lma/icon.ico"))
    manifest_path = root / "share/lma/package-manifest.json"
    manifest = {
        "schema_version": 1,
        "prefix": str(prefix),
        "platform": system,
        "version": package_version(source),
        "components": ["cli", "desktop"] if args.desktop else ["cli"],
        "backend": selected,
        "files": sorted(str(path.as_posix()) for path in [
            *(path for _, path in binaries + assets), manifest_path,
        ]),
        "runtime_libraries": (
            "llama.cpp is statically linked; no private runtime libraries are bundled "
            "and no cosmetic lib directory is created. OS libraries/frameworks and "
            "any selected CUDA or desktop runtime dependencies must be provided by the host."
        ),
        "source_registration": False,
        "root_convention": (
            "Package root represents the final prefix's parent; files start with prefix.name. "
            "Drive and parent directories are not reproduced."
            if system == "Windows" else "Package root represents /; final prefix is mirrored beneath it."
        ),
    }
    commands = build_commands(source, selected, jobs, args.desktop)
    print(f"Source: {source}\nPlatform: {system} (host only); backend: {selected}; jobs: {jobs}")
    print(f"Package directory: {destination}\nFinal prefix: {prefix}")
    print("Defaults: CLI only; jobs: 2; backend: " + ("metal" if system == "Darwin" else "cpu"))
    print(manifest["root_convention"])
    for _, relative in binaries + assets:
        final = prefix / relative.relative_to(root)
        print(f"Stage: {destination / relative} (final: {final})")
    print(f"Manifest: {destination / manifest_path} (final: {prefix / 'share/lma/package-manifest.json'})")
    print("No system installation, elevation or source registration; ~/.lma is not used.")
    if args.dry_run:
        for command in commands:
            print("Would run: " + subprocess.list2cmdline(command))
        print("Would require nonempty, regular, non-symlink licenses/assets and build outputs, "
              "then atomically publish a fresh package root. No changes made.")
        return
    for file in [source / "Cargo.lock", *([source / "desktop/Cargo.toml"] if args.desktop else [])]:
        if not file.is_file():
            raise RuntimeError(f"Missing {file}; use a complete source checkout with committed lockfiles.")
    for file, _ in assets:
        required_package_file(file)
    target = checked_absolute_path(source / "target", "Build cache")
    checked_absolute_path(target / "release", "Build output directory")
    for binary, _ in binaries:
        checked_absolute_path(binary, "Build output")
    require_tools(system, selected, False, args.desktop)
    target.mkdir(exist_ok=True)
    lock = target / ".source-install-lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"Another installer is active. Verify no build is running before removing {lock}.") from error
    try:
        for command in commands:
            run(command, source)
        # Recheck after building, before creating any package output.
        check_package_destination(destination, source, home)
        for file, _ in binaries + assets:
            required_package_file(file)
        with tempfile.TemporaryDirectory(prefix=".lma-package-", dir=destination.parent) as temporary:
            stage = Path(temporary)
            for file, relative in binaries + assets:
                output = stage / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, output)
            (stage / manifest_path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            stage.chmod(0o755)
            check_package_destination(destination, source, home)
            os.rename(stage, destination)
    finally:
        lock.rmdir()
    print(f"Completed package root: {destination}. No local installation or registration was changed.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build/install this checkout locally, or stage a host package. Never installs toolchains."
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
    parser.add_argument(
        "--package-dir", type=Path, metavar="ABSOLUTE_PATH",
        help="stage a fresh package root (must not exist; existing nonsymlink parent required); no registration/elevation",
    )
    parser.add_argument(
        "--prefix", type=Path, metavar="ABSOLUTE_PATH",
        help="final installation prefix, only with --package-dir; defaults: Linux /usr, macOS/FreeBSD /usr/local, "
             "Windows ProgramFiles/lma. POSIX stages mirror /; Windows package root represents prefix.parent "
             "(ProgramFiles/lma becomes PACKAGE_DIR/lma).",
    )
    args = parser.parse_args(argv)
    source = Path(__file__).resolve().parent.parent
    system, home = platform.system(), Path.home()
    if args.prefix is not None and args.package_dir is None:
        raise RuntimeError("--prefix is only valid with --package-dir")
    if args.package_dir is not None:
        stage_package(args, source, system, home)
        return
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
    validate_selection(system, selected, jobs, desktop)
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
