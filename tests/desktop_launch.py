"""Bounded macOS launch/UI checks using an isolated app-data directory.

Build the desktop debug binary, then run this script. --exercise also drives real
WebKit navigation, safe Markdown rendering, responsive layout, and a two-launch
workspace save/restore check. No models are loaded, installed, or downloaded.
No Accessibility or Screen Recording permission is requested.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "lma desktop: frontend IPC connected"
DISCOVERY_MARKER = "lma desktop: model discovery complete"
REPORT_MARKER = "lma desktop: native UI report "
WINDOW_PROBE = r"""
import CoreGraphics
import Foundation
let pid = Int(CommandLine.arguments[1])!
let windows = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]] ?? []
let matching = windows.filter {
    ($0[kCGWindowOwnerPID as String] as? Int) == pid &&
    ($0[kCGWindowLayer as String] as? Int) == 0
}
let result: [String: Any] = ["count": matching.count, "windows": matching.map {
    ["bounds": $0[kCGWindowBounds as String] ?? [:], "owner": $0[kCGWindowOwnerName as String] ?? ""]
}]
let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
print(String(data: data, encoding: .utf8)!)
"""


def launch(binary, environment, log, timeout, expected_phase=None):
    process = None
    try:
        with log.open("wb") as output:
            process = subprocess.Popen(
                [str(binary)],
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + timeout
            last_window = None
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Desktop exited with status {process.returncode}; see {log}"
                    )
                text = log.read_text(errors="backslashreplace")
                report = None
                for line in text.splitlines():
                    if REPORT_MARKER in line:
                        report = json.loads(line.split(REPORT_MARKER, 1)[1])
                if report is not None and not report.get("ok"):
                    raise RuntimeError(f"Native UI check failed: {report}; see {log}")
                ready = MARKER in text and DISCOVERY_MARKER in text
                if expected_phase is not None:
                    ready = ready and report is not None
                if ready:
                    if (
                        expected_phase is not None
                        and report.get("phase") != expected_phase
                    ):
                        raise RuntimeError(
                            f"Unexpected persistence phase: {report}; see {log}"
                        )
                    probe = subprocess.run(
                        ["swift", "-e", WINDOW_PROBE, str(process.pid)],
                        capture_output=True,
                        text=True,
                        timeout=15,
                        check=True,
                    )
                    last_window = json.loads(probe.stdout)
                    if last_window["count"] > 0:
                        time.sleep(0.25)
                        if process.poll() is not None:
                            raise RuntimeError(
                                "Desktop exited immediately after frontend startup"
                            )
                        print(
                            "PASS: native window, frontend IPC, and installed-model discovery"
                        )
                        print("Window:", json.dumps(last_window))
                        if report:
                            print(
                                f"PASS: {report['phase']} — {len(report['checks'])} native UI checks"
                            )
                            for check in report["checks"]:
                                print("  " + check)
                        print("Log:", log)
                        return
                time.sleep(0.2)
            raise RuntimeError(
                f"Window/IPC/UI readiness timed out; last window: {last_window}; see {log}"
            )
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            print("Test process stopped:", process.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "target/debug/lma-desktop"
    )
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument(
        "--exercise",
        action="store_true",
        help="Exercise native UI and persist across two launches",
    )
    options = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error(
            "This WindowServer probe is macOS-only; test other platforms natively."
        )
    if not 1 <= options.timeout <= 120:
        parser.error("--timeout must be 1..120 seconds per launch")
    if not shutil.which("swift"):
        parser.error("swift is required for the read-only WindowServer probe")
    binary = options.binary.resolve()
    if not binary.is_file():
        parser.error(f"Build the desktop debug binary first: {binary}")
    (ROOT / "target").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="lma-launch-", dir=ROOT / "target"
    ) as isolated:
        environment = dict(
            os.environ, LMA_DESKTOP_TEST_DIR=str(Path(isolated).resolve())
        )
        environment.pop("LMA_DESKTOP_UI_TEST", None)
        if options.exercise:
            environment["LMA_DESKTOP_UI_TEST"] = "1"
            for phase in ["written", "restored"]:
                launch(
                    binary,
                    environment,
                    ROOT / f"target/desktop-launch-{phase}.log",
                    options.timeout,
                    phase,
                )
        else:
            launch(
                binary, environment, ROOT / "target/desktop-launch.log", options.timeout
            )


if __name__ == "__main__":
    main()
