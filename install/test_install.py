"""Isolated installer tests: fake build/VCS tools, temporary checkouts and homes.
Run: python3 -B -m unittest discover -s install -p 'test_*.py' -v
"""

import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "source_install", HERE / "source_install.py"
)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)

FAKE_TOOL = """import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
with open(os.environ["FAKE_LOG"], "a") as log:
    log.write(json.dumps([name] + sys.argv[1:]) + "\\n")
if name == "ghx":
    if sys.argv[1:3] == ["remote", "list"]:
        print("origin          " + os.environ.get("FAKE_ORIGIN", "https://github.com/Hexadecimall/lma.git"))
    elif sys.argv[1] == "status":
        print(os.environ.get("FAKE_STATUS", "working tree clean"))
    elif sys.argv[1] == "pull" and os.environ.get("FAKE_PULL_FAIL"):
        sys.exit(4)
if name == "cargo":
    binary = sys.argv[sys.argv.index("--bin") + 1]
    if os.environ.get("FAKE_BUILD_FAIL") == binary:
        sys.exit(8)
    target = pathlib.Path(sys.argv[sys.argv.index("--target-dir") + 1]) / "release"
    target.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if os.environ.get("FAKE_SYSTEM") == "Windows" else ""
    output = target / (binary + suffix)
    output.write_text("new " + binary)
    output.chmod(0o755)
"""


@unittest.skipIf(
    os.name == "nt",
    "POSIX fake executable fixtures; Windows paths/staging are simulated",
)
class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lma-install-test-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.source = self.base / "source with spaces"
        self.home = self.base / "home with spaces"
        self.tools = self.base / "tools"
        self.log = self.base / "commands.jsonl"
        for directory in (
            self.source / "install",
            self.source / "desktop",
            self.home,
            self.tools,
        ):
            directory.mkdir(parents=True)
        for name in ("source_install.py", "install.sh", "Info.plist"):
            shutil.copy2(HERE / name, self.source / "install" / name)
        for name in ("Cargo.toml", "Cargo.lock", "desktop/Cargo.toml"):
            (self.source / name).write_text("fixture")
        (self.source / "Cargo.toml").write_text('[package]\nname = "lma"\nversion = "1.2.3"\n\n[dependencies]\nother = "9.9.9"\n')
        for name, content in (
            ("LICENSE", "fixture lma license"),
            ("vendor/llama-cpp-sys-2/llama.cpp/LICENSE", "fixture llama.cpp license"),
            ("desktop/icons/icon.png", "fixture png"),
            ("desktop/icons/icon.ico", "fixture ico"),
        ):
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        for tool in (
            "cargo",
            "rustc",
            "cmake",
            "cc",
            "pkg-config",
            "xcrun",
            "codesign",
            "ghx",
            "nvcc",
        ):
            path = self.tools / tool
            path.write_text("#!" + sys.executable + "\n" + FAKE_TOOL)
            path.chmod(0o755)
        self.env = {
            "HOME": str(self.home),
            "PATH": str(self.tools) + os.pathsep + os.environ.get("PATH", ""),
            "XDG_CONFIG_HOME": str(self.home / "config"),
            "XDG_DATA_HOME": str(self.home / "data"),
            "APPDATA": str(self.home / "roaming"),
            "LOCALAPPDATA": str(self.home / "local"),
            "ProgramFiles": str(self.base / "Program Files"),
            "FAKE_LOG": str(self.log),
            "FAKE_SYSTEM": "Linux",
        }
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, self.env))
        self.stack.enter_context(
            patch.object(
                installer, "__file__", str(self.source / "install/source_install.py")
            )
        )
        self.output = io.StringIO()
        self.stack.enter_context(contextlib.redirect_stdout(self.output))

    def invoke(self, *args, system="Linux"):
        with (
            patch.object(installer.platform, "system", return_value=system),
            patch.dict(os.environ, {"FAKE_SYSTEM": system}),
        ):
            installer.main(list(args))

    def commands(self):
        return (
            [json.loads(line) for line in self.log.read_text().splitlines()]
            if self.log.exists()
            else []
        )

    def snapshot(self):
        return sorted(str(p.relative_to(self.base)) for p in self.base.rglob("*"))

    def test_dry_run_writes_nothing(self):
        before = self.snapshot()
        self.invoke("--dry-run", "--cuda", "--jobs", "3")
        self.assertEqual(before, self.snapshot())
        self.assertEqual(self.commands(), [])
        self.assertIn("--locked --release --jobs 3", self.output.getvalue())

    def test_linux_build_and_registration(self):
        self.invoke()
        config_path = self.home / ".lma/source-install.json"
        config = json.loads(config_path.read_text())
        self.assertEqual(config["schema_version"], 1)
        self.assertEqual(config["source_dir"], str(self.source))
        self.assertEqual(config["backend"], "cpu")
        self.assertEqual(config["jobs"], 2)
        self.assertEqual((self.home / ".local/bin/lma").read_text(), "new lma")
        self.assertIs(config["desktop"], False)
        self.assertEqual(set(config["install_locations"]), {"cli"})
        self.assertFalse((self.home / ".local/bin/lma-desktop").exists())
        self.assertFalse((self.home / "data/applications").exists())
        commands = [c for c in self.commands() if c[0] == "cargo"]
        self.assertEqual(len(commands), 1)
        for command in commands:
            for flag in ("--locked", "--release", "--no-default-features"):
                self.assertIn(flag, command)
            self.assertEqual(
                command[command.index("--target-dir") + 1], str(self.source / "target")
            )
        self.assertIn("vision,image-gen,speech", commands[0])
        self.assertFalse(any(c[0] == "pkg-config" for c in self.commands()))

    def test_linux_desktop_opt_in(self):
        self.invoke("--desktop")
        config = json.loads((self.home / ".lma/source-install.json").read_text())
        self.assertIs(config["desktop"], True)
        self.assertEqual(set(config["install_locations"]), {"cli", "desktop", "desktop_entry"})
        self.assertEqual((self.home / ".local/bin/lma-desktop").read_text(), "new lma-desktop")
        entry = (self.home / "data/applications/org.lma.app.desktop").read_text()
        self.assertIn('Exec="' + str(self.home), entry)
        commands = [c for c in self.commands() if c[0] == "cargo"]
        self.assertEqual(len(commands), 2)
        self.assertIn("tauri/custom-protocol", commands[1])
        self.assertTrue(any(c[0] == "pkg-config" for c in self.commands()))

    def test_mac_bundle_and_metal(self):
        self.invoke("--desktop", system="Darwin")
        app = self.home / "Applications/lma.app"
        self.assertEqual(
            (app / "Contents/MacOS/lma-desktop").read_text(), "new lma-desktop"
        )
        self.assertIn("org.lma.app", (app / "Contents/Info.plist").read_text())
        self.assertTrue(
            (
                self.home
                / ".lma/source-install.json"
            ).exists()
        )
        commands = self.commands()
        self.assertIn(
            "vision,image-gen,speech,metal",
            next(c for c in commands if c[0] == "cargo"),
        )
        self.assertEqual(len([c for c in commands if c[0] == "codesign"]), 2)
        with self.assertRaisesRegex(RuntimeError, "always enables Metal"):
            self.invoke("--desktop", "--cpu", system="Darwin")

    def test_cuda_and_saved_update_settings(self):
        self.invoke("--cuda", "--jobs", "3")
        self.log.unlink()
        self.invoke("--update")
        commands = self.commands()
        self.assertIn(["ghx", "pull", "origin"], commands)
        builds = [c for c in commands if c[0] == "cargo"]
        self.assertIn("vision,image-gen,speech,cuda", builds[0])
        self.assertEqual(len(builds), 1)
        self.assertEqual(builds[0][builds[0].index("--jobs") + 1], "3")
        self.assertLess(
            commands.index(["ghx", "status"]), commands.index(["ghx", "pull", "origin"])
        )

    def test_update_dry_run_never_pulls_or_builds(self):
        self.invoke("--update", "--dry-run")
        self.assertEqual(
            self.commands(), [["ghx", "remote", "list"], ["ghx", "status"]]
        )
        self.assertFalse((self.source / "target").exists())
        self.assertEqual(list(self.home.iterdir()), [])

    def test_dirty_untracked_and_unknown_status_fail_closed(self):
        for status in (
            "unstaged   modified   Cargo.toml",
            "unstaged   new   surprise",
            "",
            "clean",
        ):
            with (
                self.subTest(status=status),
                patch.dict(os.environ, {"FAKE_STATUS": status}),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "dirty or ghx status is unrecognized"
                ):
                    self.invoke("--update")
        self.assertFalse(
            any(c[0] == "cargo" or c[:2] == ["ghx", "pull"] for c in self.commands())
        )

    def test_wrong_origin_and_pull_failure_never_build(self):
        with patch.dict(os.environ, {"FAKE_ORIGIN": "https://example.com/other.git"}):
            with self.assertRaisesRegex(RuntimeError, "origin must be exactly"):
                self.invoke("--update")
        self.assertNotIn(["ghx", "pull", "origin"], self.commands())
        with patch.dict(os.environ, {"FAKE_PULL_FAIL": "1"}):
            with self.assertRaisesRegex(RuntimeError, "ghx failed"):
                self.invoke("--update")
        self.assertFalse(any(c[0] == "cargo" for c in self.commands()))

    def test_failed_second_build_leaves_existing_install_untouched(self):
        self.invoke("--desktop")
        cli = self.home / ".local/bin/lma"
        cli.write_text("old cli")
        config = self.home / ".lma/source-install.json"
        original = config.read_bytes()
        with patch.dict(os.environ, {"FAKE_BUILD_FAIL": "lma-desktop"}):
            with self.assertRaisesRegex(RuntimeError, "cargo failed"):
                self.invoke("--desktop")
        self.assertEqual(cli.read_text(), "old cli")
        self.assertEqual(config.read_bytes(), original)
        self.assertFalse((self.source / "target/.source-install-lock").exists())

    def test_publish_failure_rolls_back_all_items(self):
        src, first, second = (
            self.base / "new",
            self.base / "first",
            self.base / "second",
        )
        src.write_text("new")
        first.write_text("old first")
        second.mkdir()
        (second / "old").write_text("old directory")
        original = os.replace

        def fail_second(source, destination):
            if Path(destination) == second and Path(source).name == "new":
                raise OSError("injected publish failure")
            return original(source, destination)

        with patch.object(installer.os, "replace", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "injected"):
                installer.publish([(src, first), (src, second)])
        self.assertEqual(first.read_text(), "old first")
        self.assertEqual((second / "old").read_text(), "old directory")
        self.assertFalse(list(self.base.glob(".lma-install-*")))

    def test_windows_locked_executable_is_staged_and_applied(self):
        self.invoke("--desktop", system="Windows")
        binary = self.home / "local/lma/bin/lma-desktop.exe"
        binary.write_text("running old desktop")
        original = os.replace

        def locked(source, destination):
            if Path(destination) == binary and Path(source).name == "new":
                raise PermissionError("running exe")
            return original(source, destination)

        with patch.object(installer.os, "replace", side_effect=locked):
            with self.assertRaisesRegex(RuntimeError, "--apply-staged"):
                self.invoke("--desktop", system="Windows")
        self.assertEqual(binary.read_text(), "running old desktop")
        self.assertTrue(
            (self.home / "local/lma/pending-update/lma-desktop.exe").exists()
        )
        self.log.unlink()
        self.invoke("--apply-staged", system="Windows")
        self.assertEqual(binary.read_text(), "new lma-desktop")
        self.assertEqual(self.commands(), [])
        self.assertFalse((self.home / "local/lma/pending-update").exists())

    def test_defaults_across_platforms_skip_desktop_checkout_and_tools(self):
        shutil.rmtree(self.source / "desktop")
        (self.source / "install/Info.plist").unlink()
        original = shutil.which
        def without_webview(tool):
            return None if tool in ("pkg-config", "codesign") else original(tool)
        with patch.object(installer.shutil, "which", side_effect=without_webview):
            for system in ("Linux", "Darwin", "Windows", "FreeBSD"):
                with self.subTest(system=system):
                    if self.log.exists():
                        self.log.unlink()
                    self.invoke(system=system)
                    builds = [c for c in self.commands() if c[0] == "cargo"]
                    self.assertEqual(len(builds), 1)
                    self.assertEqual(builds[0][builds[0].index("--bin") + 1], "lma")
                    self.assertFalse(any(c[0] in ("pkg-config", "codesign") for c in self.commands()))
                    config = json.loads((self.home / ".lma/source-install.json").read_text())
                    self.assertIs(config["desktop"], False)
                    self.assertEqual(set(config["install_locations"]), {"cli"})
                    self.assertEqual(config["backend"], "metal" if system == "Darwin" else "cpu")
                    self.assertFalse((self.home / "Applications").exists())

    def test_mac_cli_cpu_and_backend_platform_validation(self):
        self.invoke("--cpu", system="Darwin")
        builds = [c for c in self.commands() if c[0] == "cargo"]
        self.assertEqual(len(builds), 1)
        self.assertIn("vision,image-gen,speech", builds[0])
        self.assertNotIn(["xcrun", "--find", "metal"], self.commands())
        for backend in ("--cpu", "--cuda"):
            with self.assertRaisesRegex(RuntimeError, "always enables Metal"):
                self.invoke("--desktop", backend, system="Darwin")
        for system in ("Linux", "Windows"):
            with self.assertRaisesRegex(RuntimeError, "Metal is supported only on macOS"):
                self.invoke("--backend", "metal", system=system)

    def test_cli_runs_and_updates_preserve_old_desktop_and_ignore_saved_opt_in(self):
        for system in ("Linux", "Darwin", "Windows"):
            for legacy in (False, True):
                with self.subTest(system=system, legacy=legacy):
                    self.invoke("--desktop", "--jobs", "3", system=system)
                    _, paths = installer.locations(system, self.home, True)
                    if system == "Darwin":
                        old = paths["desktop"] / "Contents/MacOS/lma-desktop"
                    else:
                        old = paths["desktop"]
                    old.write_text("keep old desktop")
                    registration = self.home / ".lma/source-install.json"
                    if legacy:
                        saved = json.loads(registration.read_text())
                        del saved["desktop"]
                        registration.write_text(json.dumps(saved))
                    entry = paths.get("desktop_entry")
                    if entry:
                        entry.write_text("keep launcher")
                    for args in (("--update",), ()):
                        self.log.unlink()
                        self.invoke(*args, system=system)
                        self.assertEqual(old.read_text(), "keep old desktop")
                        if entry:
                            self.assertEqual(entry.read_text(), "keep launcher")
                        self.assertEqual(len([c for c in self.commands() if c[0] == "cargo"]), 1)
                        saved = json.loads(registration.read_text())
                        self.assertIs(saved["desktop"], False)
                        self.assertEqual(set(saved["install_locations"]), {"cli"})

    def test_desktop_can_be_opted_into_on_update(self):
        self.invoke()
        self.log.unlink()
        self.invoke("--update", "--desktop")
        self.assertEqual(len([c for c in self.commands() if c[0] == "cargo"]), 2)
        self.assertTrue((self.home / ".local/bin/lma-desktop").exists())

    def test_desktop_prerequisites_only_required_on_opt_in(self):
        original = shutil.which
        with patch.object(installer.shutil, "which", side_effect=lambda tool: None if tool == "pkg-config" else original(tool)):
            self.invoke()
            with self.assertRaisesRegex(RuntimeError, "pkg-config"):
                self.invoke("--desktop")
        (self.source / "desktop/Cargo.toml").unlink()
        self.invoke()
        with self.assertRaisesRegex(RuntimeError, "desktop/Cargo.toml"):
            self.invoke("--desktop")

    def test_registration_desktop_requires_exact_boolean(self):
        self.invoke()
        registration = self.home / ".lma/source-install.json"
        saved = json.loads(registration.read_text())
        self.log.unlink()
        for value in (None, 0, 1, "false", [], {}):
            saved["desktop"] = value
            registration.write_text(json.dumps(saved))
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "desktop must be a boolean"):
                self.invoke("--update")
        self.assertEqual(self.commands(), [])

    def test_windows_cli_staging_ignores_stale_desktop_and_apply_overrides(self):
        self.invoke("--desktop", system="Windows")
        binary = self.home / "local/lma/bin/lma.exe"
        desktop = binary.with_name("lma-desktop.exe")
        desktop.write_text("keep running desktop")
        pending = self.home / "local/lma/pending-update"
        pending.mkdir()
        (pending / "lma-desktop.exe").write_text("stale staged desktop")
        binary.write_text("running old cli")
        original = os.replace
        def locked(source, destination):
            if Path(destination) == binary and Path(source).name == "new":
                raise PermissionError("running exe")
            return original(source, destination)
        with patch.object(installer.os, "replace", side_effect=locked):
            with self.assertRaisesRegex(RuntimeError, "--apply-staged"):
                self.invoke(system="Windows")
        self.assertEqual(binary.read_text(), "running old cli")
        self.assertEqual(desktop.read_text(), "keep running desktop")
        staged = json.loads((pending / "source-install.json").read_text())
        self.assertIs(staged["desktop"], False)
        self.assertEqual(set(staged["install_locations"]), {"cli"})
        self.log.unlink()
        for args in (("--desktop",), ("--cpu",), ("--cuda",), ("--backend", "cpu"), ("--jobs", "2")):
            with self.assertRaisesRegex(RuntimeError, "do not override"):
                self.invoke("--apply-staged", *args, system="Windows")
        for value in (None, 0, 1, "false", [], {}):
            changed = dict(staged, desktop=value)
            (pending / "source-install.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(RuntimeError, "desktop must be a boolean"):
                self.invoke("--apply-staged", system="Windows")
        (pending / "source-install.json").write_text(json.dumps(staged))
        (pending / "lma-desktop.exe").unlink()
        shutil.rmtree(self.source / "desktop")
        self.invoke("--apply-staged", system="Windows")
        self.assertEqual(binary.read_text(), "new lma")
        self.assertEqual(desktop.read_text(), "keep running desktop")
        self.assertEqual(self.commands(), [])
        self.assertFalse(pending.exists())

    def test_windows_legacy_staged_registration_requires_both_binaries(self):
        self.invoke("--desktop", system="Windows")
        _, paths = installer.locations("Windows", self.home, True)
        pending = self.home / "local/lma/pending-update"
        pending.mkdir()
        registration = self.home / ".lma/source-install.json"
        saved = json.loads(registration.read_text())
        del saved["desktop"]
        (pending / "source-install.json").write_text(json.dumps(saved))
        (pending / "lma.exe").write_text("staged cli")
        with self.assertRaisesRegex(RuntimeError, "lma-desktop.exe"):
            self.invoke("--apply-staged", system="Windows")
        self.assertEqual(paths["cli"].read_text(), "new lma")
        (pending / "lma-desktop.exe").write_text("staged desktop")
        self.log.unlink()
        self.invoke("--apply-staged", system="Windows")
        self.assertEqual(paths["cli"].read_text(), "staged cli")
        self.assertEqual(paths["desktop"].read_text(), "staged desktop")
        self.assertEqual(self.commands(), [])

    def test_lock_and_invalid_jobs(self):
        lock = self.source / "target/.source-install-lock"
        lock.mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "Another installer"):
            self.invoke()
        for jobs in ("0", "65", "-1"):
            with self.assertRaisesRegex(RuntimeError, "--jobs must"):
                self.invoke("--jobs", jobs)
        self.assertTrue(lock.exists())
        self.assertFalse(any(c[0] == "cargo" for c in self.commands()))

    def test_malformed_or_redirected_registration_refused(self):
        config = self.home / ".lma/source-install.json"
        config.parent.mkdir(parents=True)
        for value in ([], {}, {"schema_version": 2, "source_dir": str(self.source)}):
            config.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "checkout/schema"):
                self.invoke("--update")
        config.write_text(" " * 65537)
        with self.assertRaisesRegex(RuntimeError, "exceeds 64 KiB"):
            self.invoke("--update")
        config.unlink()
        config.symlink_to(self.source / "Cargo.toml")
        with self.assertRaisesRegex(RuntimeError, "non-symlink"):
            self.invoke("--update")
        self.assertEqual(self.commands(), [])

    def test_package_defaults_across_hosts_and_exact_inventory(self):
        # Stale desktop artifacts and unrelated libraries must never leak into CLI packages.
        release = self.source / "target/release"
        release.mkdir(parents=True)
        for name in ("lma-desktop", "lma-desktop.exe", "libllama.dylib", "random.dll"):
            (release / name).write_text("stale")
        original_source = {
            p.relative_to(self.source): p.read_bytes()
            for p in self.source.rglob("*") if p.is_file() and "target" not in p.relative_to(self.source).parts
        }
        for system in ("Linux", "Darwin", "FreeBSD", "Windows"):
            with self.subTest(system=system):
                if self.log.exists():
                    self.log.unlink()
                stage = self.base / ("package " + system)
                self.invoke("--package-dir", str(stage), system=system)
                root = stage / ("lma" if system == "Windows" else "usr" if system == "Linux" else "usr/local")
                suffix = ".exe" if system == "Windows" else ""
                self.assertEqual((root / ("bin/lma" + suffix)).read_text(), "new lma")
                self.assertEqual((root / "share/lma/appicon.png").read_text(), "fixture png")
                self.assertEqual((root / "share/lma/icon.ico").read_text(), "fixture ico")
                self.assertEqual((root / "share/licenses/lma/LICENSE").read_text(), "fixture lma license")
                self.assertEqual((root / "share/licenses/lma/llama.cpp/LICENSE").read_text(), "fixture llama.cpp license")
                manifest = json.loads((root / "share/lma/package-manifest.json").read_text())
                expected_prefix = str(Path(os.environ["ProgramFiles"]) / "lma") if system == "Windows" else "/usr" if system == "Linux" else "/usr/local"
                self.assertEqual(manifest["prefix"], expected_prefix)
                self.assertEqual(manifest["platform"], system)
                self.assertEqual(manifest["version"], "1.2.3")
                self.assertEqual(manifest["components"], ["cli"])
                self.assertEqual(manifest["backend"], "metal" if system == "Darwin" else "cpu")
                self.assertIs(manifest["source_registration"], False)
                self.assertNotIn("source_dir", manifest)
                self.assertIn("statically linked", manifest["runtime_libraries"])
                actual = sorted(p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file())
                expected = sorted((root / name).relative_to(stage).as_posix() for name in (
                    "bin/lma" + suffix, "share/lma/appicon.png", "share/lma/icon.ico",
                    "share/licenses/lma/LICENSE", "share/licenses/lma/llama.cpp/LICENSE",
                    "share/lma/package-manifest.json",
                ))
                self.assertEqual(actual, expected)
                self.assertEqual(manifest["files"], expected)
                self.assertFalse(any(p.name in ("lib", "lma.app", installer.CONFIG_NAME) for p in stage.rglob("*")))
                self.assertEqual(list(self.home.iterdir()), [])
                builds = [c for c in self.commands() if c[0] == "cargo"]
                self.assertEqual(builds, installer.build_commands(self.source, manifest["backend"], 2))
                self.assertFalse(any(c[0] in ("ghx", "codesign", "pkg-config") for c in self.commands()))
                self.assertFalse((self.source / "target/.source-install-lock").exists())
        self.assertEqual(original_source, {
            p.relative_to(self.source): p.read_bytes()
            for p in self.source.rglob("*") if p.is_file() and "target" not in p.relative_to(self.source).parts
        })

    def test_package_desktop_is_plain_binary_on_every_supported_host(self):
        (self.source / "install/Info.plist").unlink()
        for system in ("Linux", "Darwin", "Windows"):
            with self.subTest(system=system):
                stage = self.base / ("desktop-package-" + system)
                self.invoke("--package-dir", str(stage), "--desktop", system=system)
                root = stage / ("lma" if system == "Windows" else "usr" if system == "Linux" else "usr/local")
                suffix = ".exe" if system == "Windows" else ""
                self.assertEqual((root / ("bin/lma-desktop" + suffix)).read_text(), "new lma-desktop")
                self.assertEqual(json.loads((root / "share/lma/package-manifest.json").read_text())["components"], ["cli", "desktop"])
                self.assertFalse(any(p.suffix in (".app", ".desktop") for p in stage.rglob("*")))
                self.assertEqual(list(self.home.iterdir()), [])
        self.assertFalse(any(c[0] == "codesign" for c in self.commands()))

    def test_package_prefix_override_and_no_local_environment_dependency(self):
        for system in ("Linux", "Darwin", "FreeBSD", "Windows"):
            with self.subTest(system=system), patch.dict(os.environ, {"LOCALAPPDATA": "", "XDG_DATA_HOME": "relative", "ProgramFiles": ""}):
                stage = self.base / ("custom-" + system)
                prefix = self.base / "final tree/custom-lma"
                self.invoke("--package-dir", str(stage), "--prefix", str(prefix), system=system)
                root = stage / (prefix.name if system == "Windows" else prefix.relative_to(prefix.anchor))
                suffix = ".exe" if system == "Windows" else ""
                self.assertTrue((root / ("bin/lma" + suffix)).is_file())
                self.assertFalse(prefix.exists())
                self.assertEqual(json.loads((root / "share/lma/package-manifest.json").read_text())["prefix"], str(prefix))
        self.assertEqual(list(self.home.iterdir()), [])

    def test_package_windows_requires_program_files_only_for_default(self):
        with patch.dict(os.environ, {"ProgramFiles": ""}):
            with self.assertRaisesRegex(RuntimeError, "ProgramFiles must be set"):
                self.invoke("--package-dir", str(self.base / "stage"), "--dry-run", system="Windows")
        self.assertEqual(self.commands(), [])

    def test_package_dry_run_missing_licenses_and_outputs_writes_nothing(self):
        (self.source / "LICENSE").unlink()
        (self.source / "vendor/llama-cpp-sys-2/llama.cpp/LICENSE").unlink()
        (self.source / "desktop/icons/icon.png").unlink()
        before = self.snapshot()
        with patch.object(installer, "require_tools", side_effect=AssertionError("must not check tools")):
            for system in ("Linux", "Darwin", "FreeBSD", "Windows"):
                with self.subTest(system=system):
                    self.output.seek(0)
                    self.output.truncate()
                    stage = self.base / ("dry-" + system)
                    self.invoke("--package-dir", str(stage), "--dry-run", system=system)
                    text = self.output.getvalue()
                    root = stage / ("lma" if system == "Windows" else "usr" if system == "Linux" else "usr/local")
                    self.assertIn(str(root / "share/licenses/lma/LICENSE"), text)
                    self.assertIn(str(root / "share/licenses/lma/llama.cpp/LICENSE"), text)
                    self.assertIn(str(root / "share/lma/appicon.png"), text)
                    self.assertIn(str(root / "share/lma/package-manifest.json"), text)
                    self.assertIn(str(root / ("bin/lma.exe" if system == "Windows" else "bin/lma")), text)
                    self.assertIn("Defaults: CLI only; jobs: 2; backend:", text)
                    self.assertIn("--locked --release --jobs 2", text)
                    self.assertNotIn("lma-desktop", text)
                    self.assertIn("No changes made", text)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(self.commands(), [])

    def test_package_license_and_asset_requirements_fail_before_build(self):
        for name in ("LICENSE", "vendor/llama-cpp-sys-2/llama.cpp/LICENSE", "desktop/icons/icon.png", "desktop/icons/icon.ico"):
            file = self.source / name
            content = file.read_bytes()
            for kind in ("missing", "empty", "directory", "symlink"):
                # The second inventoried image is optional, but must be safe if present.
                if name.endswith("icon.ico") and kind == "missing":
                    continue
                with self.subTest(name=name, kind=kind):
                    file.unlink()
                    if kind == "empty":
                        file.touch()
                    elif kind == "directory":
                        file.mkdir()
                    elif kind == "symlink":
                        file.symlink_to(self.source / "Cargo.toml")
                    try:
                        with self.assertRaisesRegex(RuntimeError, "nonempty regular|symlink"):
                            self.invoke("--package-dir", str(self.base / "bad-license"))
                        self.assertFalse((self.base / "bad-license").exists())
                        self.assertFalse((self.source / "target").exists())
                        self.assertEqual(self.commands(), [])
                    finally:
                        installer.remove(file)
                        file.write_bytes(content)

    def test_package_optional_icon_absent_and_symlink_asset_parent(self):
        (self.source / "desktop/icons/icon.ico").unlink()
        stage = self.base / "one-icon"
        self.invoke("--package-dir", str(stage))
        self.assertFalse((stage / "usr/share/lma/icon.ico").exists())
        icons = self.source / "desktop/icons"
        moved = self.base / "moved-icons"
        icons.rename(moved)
        icons.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            self.invoke("--package-dir", str(self.base / "linked-assets"))

    def test_package_rejects_existing_unsafe_and_overlapping_destinations(self):
        existing = self.base / "existing"
        existing.mkdir()
        (existing / "keep").write_text("untouched")
        empty = self.base / "empty"
        empty.mkdir()
        file = self.base / "file"
        file.write_text("keep file")
        link = self.base / "link"
        link.symlink_to(existing, target_is_directory=True)
        dangling = self.base / "dangling"
        dangling.symlink_to(self.base / "missing")
        for destination in (
            Path("relative"), self.base / ".." / "escape", Path("/"),
            existing, empty, file, link, dangling, link / "new", dangling / "new",
            self.source, self.source / "new", self.source / "target/new",
            self.home, self.base, self.home / ".lma/new", self.base / "missing-parent/new",
        ):
            with self.subTest(destination=destination):
                before = self.snapshot()
                with self.assertRaises(RuntimeError):
                    self.invoke("--package-dir", str(destination))
                self.assertEqual(before, self.snapshot())
        self.assertEqual((existing / "keep").read_text(), "untouched")
        self.assertEqual(file.read_text(), "keep file")
        self.assertEqual(self.commands(), [])

    def test_package_prefix_validation_and_mode_conflicts(self):
        link = self.base / "prefix-link"
        link.symlink_to(self.base, target_is_directory=True)
        file = self.base / "prefix-file"
        file.write_text("not a directory")
        for prefix in ("relative", "/", str(self.base / ".." / "escape"), str(link), str(link / "child"), str(file), str(file / "child")):
            with self.subTest(prefix=prefix), self.assertRaisesRegex(RuntimeError, "--prefix"):
                self.invoke("--package-dir", str(self.base / "stage"), "--prefix", prefix)
        with self.assertRaisesRegex(RuntimeError, "only valid with --package-dir"):
            self.invoke("--prefix", "/usr")
        for args in (("--update",), ("--apply-staged",), ("--config-dir", str(self.home / "config"))):
            with self.subTest(args=args), self.assertRaisesRegex(RuntimeError, "cannot be combined"):
                self.invoke("--package-dir", str(self.base / "stage"), *args)
        self.assertEqual(self.commands(), [])
        self.assertFalse((self.source / "target").exists())

    def test_package_failures_leave_local_install_and_package_parent_untouched(self):
        self.invoke("--desktop")
        home_before = {p.relative_to(self.home): p.read_bytes() for p in self.home.rglob("*") if p.is_file()}
        stage = self.base / "failed-package"
        original_copy = shutil.copy2

        def fail_copy(source, destination, *args, **kwargs):
            original_copy(source, destination, *args, **kwargs)
            if Path(source).name == "icon.png":
                raise OSError("injected copy failure")

        for failure in ("build", "copy", "rename", "manifest"):
            with self.subTest(failure=failure), contextlib.ExitStack() as stack:
                if failure == "build":
                    stack.enter_context(patch.dict(os.environ, {"FAKE_BUILD_FAIL": "lma-desktop"}))
                elif failure == "copy":
                    stack.enter_context(patch.object(installer.shutil, "copy2", side_effect=fail_copy))
                elif failure == "rename":
                    stack.enter_context(patch.object(installer.os, "rename", side_effect=OSError("injected rename failure")))
                else:
                    original_write = Path.write_text

                    def fail_manifest(path, *args, **kwargs):
                        if path.name == "package-manifest.json":
                            raise OSError("injected manifest failure")
                        return original_write(path, *args, **kwargs)

                    stack.enter_context(patch.object(Path, "write_text", fail_manifest))
                with self.assertRaises((OSError, RuntimeError)):
                    self.invoke("--package-dir", str(stage), "--desktop")
                self.assertFalse(stage.exists())
                self.assertFalse(list(self.base.glob(".lma-package-*")))
                self.assertFalse((self.source / "target/.source-install-lock").exists())
                self.assertEqual(home_before, {p.relative_to(self.home): p.read_bytes() for p in self.home.rglob("*") if p.is_file()})

    def test_package_destination_created_during_copy_is_not_replaced(self):
        stage = self.base / "raced-package"
        original = shutil.copy2

        def create_destination(source, destination, *args, **kwargs):
            if not stage.exists():
                stage.mkdir()
                (stage / "keep").write_text("other process")
            return original(source, destination, *args, **kwargs)

        with patch.object(installer.shutil, "copy2", side_effect=create_destination):
            with self.assertRaisesRegex(RuntimeError, "must not exist"):
                self.invoke("--package-dir", str(stage))
        self.assertEqual((stage / "keep").read_text(), "other process")
        self.assertFalse(list(self.base.glob(".lma-package-*")))

    def test_package_does_not_read_or_modify_local_registration(self):
        registration = self.home / ".lma/source-install.json"
        registration.parent.mkdir()
        registration.write_text("invalid local registration is irrelevant")
        before = registration.read_bytes()
        self.invoke("--package-dir", str(self.base / "unregistered"), system="Windows")
        self.assertEqual(registration.read_bytes(), before)
        self.assertEqual(list(self.home.rglob("*.json")), [registration])
        self.assertFalse((self.home / "local").exists())

    def test_package_target_symlink_and_active_lock_refused(self):
        target = self.source / "target"
        target.symlink_to(self.home, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            self.invoke("--package-dir", str(self.base / "stage"))
        target.unlink()
        lock = target / ".source-install-lock"
        lock.mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "Another installer"):
            self.invoke("--package-dir", str(self.base / "stage"))
        self.assertTrue(lock.exists())
        self.assertEqual(self.commands(), [])
        self.assertEqual(list(self.home.iterdir()), [])

    def test_package_invalid_build_outputs_are_never_published(self):
        original = installer.run
        binary = self.source / "target/release/lma"
        license_before = (self.source / "LICENSE").read_bytes()
        for kind in ("missing", "empty", "symlink"):
            def bad_output(command, cwd, capture=False):
                result = original(command, cwd, capture)
                if command[0] == "cargo":
                    binary.unlink()
                    if kind == "empty":
                        binary.touch()
                    elif kind == "symlink":
                        binary.symlink_to(self.source / "LICENSE")
                return result

            with self.subTest(kind=kind), patch.object(installer, "run", side_effect=bad_output):
                with self.assertRaisesRegex(RuntimeError, "nonempty regular|symlink"):
                    self.invoke("--package-dir", str(self.base / "bad-output"))
                self.assertFalse((self.base / "bad-output").exists())
                self.assertFalse((self.source / "target/.source-install-lock").exists())
                self.assertFalse(list(self.base.glob(".lma-package-*")))
                self.assertEqual((self.source / "LICENSE").read_bytes(), license_before)
                installer.remove(binary)

    def test_package_preexisting_binary_symlink_refused_before_build(self):
        binary = self.source / "target/release/lma"
        binary.parent.mkdir(parents=True)
        binary.symlink_to(self.source / "LICENSE")
        before = (self.source / "LICENSE").read_bytes()
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            self.invoke("--package-dir", str(self.base / "stage"))
        self.assertEqual(self.commands(), [])
        self.assertEqual((self.source / "LICENSE").read_bytes(), before)

    def test_package_desktop_dry_run_has_all_binary_destinations(self):
        (self.source / "LICENSE").write_text("")
        before = self.snapshot()
        self.invoke("--package-dir", str(self.base / "desktop-preview"), "--desktop", "--dry-run", system="Darwin")
        text = self.output.getvalue()
        for name in ("lma", "lma-desktop"):
            self.assertIn(str(self.base / "desktop-preview/usr/local/bin" / name), text)
            self.assertIn(f"(final: /usr/local/bin/{name})", text)
        self.assertNotIn("lma.app", text)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(self.commands(), [])
        self.assertEqual((self.source / "LICENSE").read_bytes(), b"")

    def test_package_source_case_alias_refused_when_filesystem_allows_it(self):
        alias = self.source.with_name(self.source.name.upper())
        if not alias.exists() or not alias.samefile(self.source):
            self.skipTest("case-sensitive filesystem")
        with self.assertRaisesRegex(RuntimeError, "overlaps protected"):
            self.invoke("--package-dir", str(alias / "new-package"))
        self.assertEqual(self.commands(), [])

    def test_freebsd_is_experimental_cpu_cli_only_for_both_modes(self):
        for mode in ([], ["--package-dir", str(self.base / "freebsd")]):
            for options in (["--desktop"], ["--cuda"], ["--backend", "metal"]):
                with self.subTest(mode=mode, options=options), self.assertRaisesRegex(RuntimeError, "CPU CLI only"):
                    self.invoke(*mode, *options, system="FreeBSD")
        self.assertEqual(self.commands(), [])
        self.invoke(system="FreeBSD")
        self.assertEqual((self.home / ".local/bin/lma").read_text(), "new lma")
        self.assertIn("experimental", self.output.getvalue())

    def test_symlink_destination_refused(self):
        src, dest = self.base / "src", self.base / "dest"
        src.write_text("keep")
        dest.symlink_to(src)
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            installer.publish([(src, dest)])
        self.assertEqual(src.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
