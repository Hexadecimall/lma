#!/usr/bin/env python3
"""Bounded smoke tests for the real lma CLI; requires an already-installed model.

Run: python3 tests/tui_launch.py
No downloads, HTTP servers, third-party Python packages, or all-GPU loads.
Each PTY has a single deadline (at most 90 seconds), including model loading.
Raw output and plain-text screen checkpoints are retained beneath target/.
"""

import argparse
import codecs
import errno
import fcntl
import json
import os
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import termios
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "hf.co/mradermacher/Pulsar-16B-BF16-GGUF:Q4_K_M"
FLAGS = [
    "-g",
    "1",
    "-c",
    "256",
    "-b",
    "32",
    "-mh",
    "32",
    "-t",
    "6",
    "-mk",
    "error",
]
EXIT_MARKER = "__LMA_TUI_EXIT__"
OUTPUT_LIMIT = 8 * 1024 * 1024
ROWS, COLS = 36, 160


class SmokeFailure(Exception):
    pass


def require(condition, message):
    if not condition:
        raise SmokeFailure(message)


def child_session():
    os.setsid()
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)


def child_wrapper(command):
    """Keep the controlling PTY alive to inspect termios after the CLI exits.

    macOS revokes a controlling terminal when its session leader exits. This
    small supervisor owns the session, but lma itself receives the real PTY.
    The outer test always terminates/kills the entire process group on failure.
    """
    before = termios.tcgetattr(0)
    try:
        code = subprocess.run(command, timeout=90, check=False).returncode
    except subprocess.TimeoutExpired:
        code = 124
    after = termios.tcgetattr(0)
    # BSD may set this transient retype-input flag when leaving raw mode.
    before[3] &= ~getattr(termios, "PENDIN", 0)
    after[3] &= ~getattr(termios, "PENDIN", 0)
    report = {"code": code, "termios_restored": before == after}
    print("\n" + EXIT_MARKER + json.dumps(report), flush=True)
    return 0 if code == 0 and report["termios_restored"] else 1


class Screen:
    """Small VT reader for Ratatui's cursor-addressed, non-scrolling frames.

    Reconstructing cells matters: later frames are diffs, not full text writes.
    It also answers cursor-position probes without requiring a terminal emulator.
    This is test infrastructure, not an interpreter for model escape sequences.
    """

    def __init__(self, send):
        self.send = send
        self.cells = [[" "] * COLS for _ in range(ROWS)]
        self.row = self.col = 0
        self.saved = (0, 0)
        self.state = "text"
        self.sequence = ""
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.enters = self.leaves = self.shows = 0
        self.last_alternate = ""
        self.mouse_modes = set()
        self.mouse_enables = 0
        self.keyboard_pushes = self.keyboard_pops = 0

    @property
    def text(self):
        return "\n".join("".join(row).rstrip() for row in self.cells)

    def feed(self, data):
        for char in self.decoder.decode(data):
            if self.state == "escape":
                self.state = {
                    "[": "csi",
                    "]": "osc",
                    "(": "charset",
                    ")": "charset",
                }.get(char, "text")
                self.sequence = ""
                if char == "7":
                    self.saved = (self.row, self.col)
                elif char == "8":
                    self.row, self.col = self.saved
            elif self.state == "charset":
                self.state = "text"
            elif self.state == "osc":
                if char == "\x07":
                    self.state = "text"
                elif char == "\x1b":
                    self.state = "osc_escape"
            elif self.state == "osc_escape":
                self.state = "text" if char == "\\" else "osc"
            elif self.state == "csi":
                if "@" <= char <= "~":
                    self.csi(self.sequence, char)
                    self.state = "text"
                else:
                    self.sequence += char
                    require(
                        len(self.sequence) <= 128, "overlong terminal control sequence"
                    )
            elif char == "\x1b":
                self.state = "escape"
            elif char == "\r":
                self.col = 0
            elif char == "\n":
                if self.row == ROWS - 1:
                    self.cells.pop(0)
                    self.cells.append([" "] * COLS)
                else:
                    self.row += 1
            elif char == "\b":
                self.col = max(0, self.col - 1)
            elif char == "\t":
                self.col = min(COLS - 1, (self.col // 8 + 1) * 8)
            elif not unicodedata.category(char).startswith(
                "C"
            ) and not unicodedata.combining(char):
                width = 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
                if self.col >= COLS:
                    self.col = 0
                    self.row = min(ROWS - 1, self.row + 1)
                self.cells[self.row][self.col] = char
                if width == 2 and self.col + 1 < COLS:
                    self.cells[self.row][self.col + 1] = ""
                self.col += width

    def csi(self, sequence, command):
        if command == "u" and sequence.startswith(">"):
            self.keyboard_pushes += 1
            return
        if command == "u" and sequence.startswith("<"):
            self.keyboard_pops += 1
            return
        if sequence == "?" and command == "u":
            self.send(b"\x1b[?1u")  # Report Kitty disambiguation support.
            return
        if not sequence and command == "c":
            self.send(b"\x1b[?1;2c")
            return
        private = sequence.startswith("?")
        numbers = sequence.lstrip("?<>=!").split(";")
        values = [int(value) if value.isdigit() else 0 for value in numbers]
        amount = values[0] or 1
        if private:
            for mode in values:
                if mode in (1000, 1002, 1003, 1006, 1015):
                    if command == "h":
                        self.mouse_modes.add(mode)
                        self.mouse_enables += 1
                    elif command == "l":
                        self.mouse_modes.discard(mode)
            if 1049 in values and command == "h":
                self.enters += 1
                self.cells = [[" "] * COLS for _ in range(ROWS)]
                self.row = self.col = 0
            elif 1049 in values and command == "l":
                self.leaves += 1
                self.last_alternate = self.text
            if 25 in values and command == "h":
                self.shows += 1
            return
        if command in ("H", "f"):
            self.row = min(ROWS - 1, amount - 1)
            self.col = min(COLS - 1, (values[1] or 1) - 1 if len(values) > 1 else 0)
        elif command == "A":
            self.row = max(0, self.row - amount)
        elif command == "B":
            self.row = min(ROWS - 1, self.row + amount)
        elif command == "C":
            self.col = min(COLS - 1, self.col + amount)
        elif command == "D":
            self.col = max(0, self.col - amount)
        elif command == "G":
            self.col = min(COLS - 1, amount - 1)
        elif command == "d":
            self.row = min(ROWS - 1, amount - 1)
        elif command == "J":
            if values[0] in (2, 3):
                self.cells = [[" "] * COLS for _ in range(ROWS)]
            elif values[0] == 0:
                self.cells[self.row][self.col :] = [" "] * (COLS - self.col)
                for row in range(self.row + 1, ROWS):
                    self.cells[row] = [" "] * COLS
        elif command == "K":
            start = 0 if values[0] in (1, 2) else min(self.col, COLS)
            end = min(self.col + 1, COLS) if values[0] == 1 else COLS
            self.cells[self.row][start:end] = [" "] * (end - start)
        elif command == "s":
            self.saved = (self.row, self.col)
        elif command == "u":
            self.row, self.col = self.saved
        elif command == "n" and values[0] == 6:
            self.send(f"\x1b[{self.row + 1};{self.col + 1}R".encode("ascii"))


class PtyRun:
    def __init__(self, binary, arguments, timeout, folder, name):
        self.command = [str(binary), *arguments]
        self.timeout = timeout
        self.folder = folder
        self.name = name
        self.process = None
        self.master = self.slave = None
        self.log = None
        self.raw = bytearray()
        self.checkpoints = []
        self.report = None
        self.screen = Screen(self.send)

    def __enter__(self):
        self.started = time.monotonic()
        self.deadline = self.started + self.timeout
        try:
            self.master, self.slave = pty.openpty()
            fcntl.ioctl(
                self.slave, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0)
            )
            self.log = (self.folder / (self.name + ".ansi.log")).open("wb")
            environment = dict(os.environ, TERM="xterm-256color")
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--pty-child",
                    *self.command,
                ],
                stdin=self.slave,
                stdout=self.slave,
                stderr=self.slave,
                cwd=ROOT,
                env=environment,
                preexec_fn=child_session,
                close_fds=True,
            )
            return self
        except BaseException:
            self.close()
            raise

    def send(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        os.write(self.master, data)

    def pump(self):
        if time.monotonic() >= self.deadline:
            raise SmokeFailure(
                f"{self.name}: timed out after {self.timeout:g}s (no automatic retry)"
            )
        ready, _, _ = select.select([self.master], [], [], 0.025)
        if ready:
            try:
                data = os.read(self.master, 65536)
            except OSError as error:
                if error.errno != errno.EIO:
                    raise
                data = b""
            require(
                len(self.raw) + len(data) <= OUTPUT_LIMIT,
                f"{self.name}: output exceeded 8 MiB",
            )
            self.raw.extend(data)
            self.log.write(data)
            self.screen.feed(data)
            marker = re.search(rb"__LMA_TUI_EXIT__(\{[^\r\n]+\})", self.raw)
            if marker:
                self.report = json.loads(marker[1])

    def wait_for(self, description, predicate):
        while True:
            self.pump()
            if predicate(self.screen.text):
                self.checkpoint(description)
                return
            engine_error = re.search(r"Engine [45]\d\d: ([^\n]+)", self.screen.text)
            if engine_error:
                raise SmokeFailure(f"{self.name}: {engine_error[0]}")
            if self.process.poll() is not None:
                raise SmokeFailure(
                    f"{self.name}: CLI exited before {description}; see {self.folder}"
                )

    def checkpoint(self, description):
        self.checkpoints.append(
            f"=== {description} ({time.monotonic() - self.started:.2f}s) ===\n{self.screen.text}"
        )

    def finish(self):
        while self.process.poll() is None or self.report is None:
            self.pump()
            if self.process.poll() is not None and self.report is None:
                raise SmokeFailure(
                    f"{self.name}: supervisor exited without terminal-restoration report"
                )
        require(
            self.report["code"] == 0,
            f"{self.name}: CLI exit code {self.report['code']}",
        )
        require(
            self.report["termios_restored"], f"{self.name}: termios was not restored"
        )
        require(
            self.screen.enters == self.screen.leaves == 1,
            f"{self.name}: alternate-screen enter/leave were not paired",
        )
        require(self.screen.shows > 0, f"{self.name}: cursor was not restored")
        require(self.screen.mouse_enables > 0 and not self.screen.mouse_modes,
                f"{self.name}: mouse capture was not enabled/restored")
        require(self.screen.keyboard_pushes == self.screen.keyboard_pops == 1,
                f"{self.name}: enhanced keyboard mode was not enabled/restored")
        return round(time.monotonic() - self.started, 2)

    def close(self):
        try:
            if self.process is not None:
                # Let lma restore its terminal and tear down native state first.
                if self.process.poll() is None:
                    try:
                        self.send(b"\x11")
                        self.process.wait(timeout=1)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
                # A normal supervisor exit has already reaped its CLI child.
                if self.process.poll() is not None and self.process.returncode >= 0:
                    return
                # The supervisor might have exited unexpectedly while its CLI
                # child remains. Clean the owned group even in that case.
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
                finally:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.process.wait(timeout=2)
        finally:
            if self.log is not None:
                self.log.close()
            for fd in (self.master, self.slave):
                if fd is not None:
                    os.close(fd)
            self.master = self.slave = None

    def __exit__(self, kind, error, traceback):
        try:
            self.close()
        finally:
            if hasattr(self, "started"):
                text = "\n\n".join(self.checkpoints)
                text += (
                    "\n\n=== last alternate screen ===\n" + self.screen.last_alternate
                )
                text += "\n\n=== last observed screen ===\n" + self.screen.text
                (self.folder / (self.name + ".txt")).write_text(text, encoding="utf-8")
                (self.folder / (self.name + ".json")).write_text(
                    json.dumps(
                        {
                            "command": self.command,
                            "elapsed_seconds": round(
                                time.monotonic() - self.started, 2
                            ),
                            "terminal": self.report,
                            "alternate_screen_enters": self.screen.enters,
                            "alternate_screen_leaves": self.screen.leaves,
                            "cursor_shows": self.screen.shows,
                            "supervisor_pid": self.process.pid
                            if self.process
                            else None,
                            "supervisor_returncode": self.process.returncode
                            if self.process
                            else None,
                            "error": str(error) if error else None,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )


def setup_smoke(binary, timeout, folder, command):
    with PtyRun(binary, [command], timeout, folder, "setup-" + command) as run:
        run.wait_for(
            "setup picker",
            lambda text: "Choose settings before loading weights" in text,
        )
        setup_text = run.screen.text
        for label in ("Model", "GPU", "Context", "Start"):
            require(
                re.search(r"(?:^|\n)[ ›]*" + label + r"\s", setup_text),
                "missing setup row: " + label,
            )
        require(
            re.search(r"(?:^|\n)[ ›]*(Mode|API|Client)\s", setup_text) is None,
            "chat setup unexpectedly exposes hosting options or a Mode row",
        )
        run.send("\r")
        run.wait_for(
            "installed-model list",
            lambda text: "Installed models" in text and "Enter choose" in text,
        )
        require(
            "Edit:" not in run.screen.text,
            "Model opened a text editor instead of a list",
        )
        require(
            "No installed models found" in run.screen.text
            or re.search(r"\[(?:lma|Ollama|LM Studio)\]", run.screen.text),
            "model picker has no source labels or empty-state explanation",
        )
        run.send(b"\x1b[B\x1b[A")  # Navigate the list without loading anything.
        run.send(b"\x1b")
        run.wait_for("back to setup", lambda text: "Enter browse/edit/start" in text)
        run.send(b"\t\t\r")  # Context is the third chat-setup row.
        run.wait_for(
            "context editor", lambda text: "Edit:" in text and "Enter apply" in text
        )
        run.send(b"\x11")  # Ctrl+Q also cancels from a numeric editor.
        elapsed = run.finish()
        require(
            b"Loading " not in run.raw and b"Backend request:" not in run.raw,
            "setup unexpectedly loaded an engine",
        )
        return {
            "case": command + " model-list/context-edit/cancel",
            "seconds": elapsed,
            "terminal_restored": True,
        }


def non_tty_smoke(binary, timeout, folder, model, local=False):
    for name, arguments in [
        ("launch", ["lh"]),
        ("chat", ["c", "-p" if local else "-o", model, *FLAGS]),
    ]:
        result = subprocess.run(
            [str(binary), *arguments],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=min(timeout, 5),
            check=False,
        )
        output = result.stdout + result.stderr
        (folder / ("non-tty-" + name + ".log")).write_bytes(output)
        require(
            result.returncode != 0 and b"needs a terminal" in output.lower(),
            name + ": missing friendly non-TTY error",
        )
        require(
            b"\x1b[?1049h" not in output and b"Loading " not in output,
            name + ": non-TTY path entered the UI or loaded weights",
        )
    return {"case": "non-TTY launch/chat", "passed": True}


def assistant_reply(text):
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "Assistant":
            reply = []
            for following in lines[index + 1 :]:
                if following.startswith("┌"):
                    break
                reply.append(following.strip())
            return "\n".join(reply).strip()
    return ""


def chat_smoke(binary, timeout, folder, model, local=False):
    with PtyRun(
        binary,
        ["c", "-p" if local else "-o", model, *FLAGS],
        timeout,
        folder,
        "real-chat",
    ) as run:
        run.wait_for(
            "native chat ready",
            lambda text: (
                "Ready" in text and "Conversation" in text and "Message" in text
            ),
        )
        run.send("DraftLineOne")
        run.send(b"\x1b[13;2u")  # Actual enhanced Shift+Enter, not Alt+Enter.
        run.send("DraftLineTwo")
        run.wait_for("Shift+Enter inserts a newline without sending",
                     lambda text: "DraftLineOne" in text and "DraftLineTwo" in text)
        require("Generating" not in run.screen.text, "Shift+Enter sent the draft")
        run.send(b"\x15\x7f\x15")  # Clear line two, newline, then line one.
        run.send("/tokens 4\r")
        run.wait_for(
            "four-token output limit",
            lambda text: "Reply limit: 4 tokens" in text and "reply limit 4 tokens" in text,
        )
        run.send("/temperature 0\r")
        run.wait_for(
            "greedy sampling",
            lambda text: "Temperature: 0" in text and "temp 0 ·" in text,
        )
        run.send("Say hello.\r")
        run.wait_for(
            "actual reply within the four-token output budget",
            lambda text: (
                bool(assistant_reply(text))
                and re.search(r"\b[1-9]\d* in / [1-4] out tokens\b", text) is not None
            ),
        )
        reply = assistant_reply(run.screen.text)
        usage = re.search(r"\b([1-9]\d*) in / ([1-4]) out tokens\b", run.screen.text)
        run.send("/tokens off\r")
        run.wait_for("reply cap disabled without unloading",
                     lambda text: "Reply limit: off" in text and "remaining context" in text)
        run.send("/help\r")
        run.wait_for("chat help panel", lambda text: "Keys" in text and "Shift+Enter" in text)
        before_scroll = run.screen.text
        run.send(b"\x1b[<65;10;10M" * 3)
        run.wait_for("SGR mouse wheel scrolls chat history", lambda text: text != before_scroll)
        before_scroll = run.screen.text
        run.send(b"\x1b[6~")
        run.wait_for("PageDown scrolls chat history", lambda text: text != before_scroll)
        run.send(b"\x1b")
        run.wait_for("back to conversation", lambda text: "Conversation" in text)
        run.send(b"\x11")
        elapsed = run.finish()
        return {
            "case": "installed model native chat",
            "model": model,
            "reply": reply,
            "prompt_tokens": int(usage[1]),
            "completion_tokens": int(usage[2]),
            "seconds": elapsed,
            "terminal_restored": True,
            "shift_enter_newline": True,
            "mouse_and_page_scrolling": True,
            "unlimited_reply_setting": True,
        }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--binary", type=Path, default=ROOT / "target/debug/lma")
    parser.add_argument(
        "--model",
        default=MODEL,
        help="already-installed Ollama reference; never downloaded",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="interpret --model as a local GGUF path instead of an Ollama reference",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90,
        help="one deadline per PTY, 1–90 seconds (default: 90)",
    )
    args = parser.parse_args()
    if not 1 <= args.timeout <= 90:
        parser.error("--timeout must be between 1 and 90 seconds")
    binary = args.binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error("--binary must name an existing executable")
    folder = (
        ROOT
        / "target"
        / ("tui-launch-" + time.strftime("%Y%m%d-%H%M%S") + "-" + str(os.getpid()))
    )
    folder.mkdir(parents=True, exist_ok=False)
    results = []
    try:
        results.append(
            non_tty_smoke(binary, args.timeout, folder, args.model, args.local)
        )
        for command in ("lh",):
            results.append(setup_smoke(binary, args.timeout, folder, command))
        # Strictly sequential: only this case loads weights, and it fully exits.
        results.append(chat_smoke(binary, args.timeout, folder, args.model, args.local))
    except (SmokeFailure, subprocess.TimeoutExpired, OSError) as error:
        for result in results:
            print("PASS: " + json.dumps(result, ensure_ascii=True), flush=True)
        print("FAIL: " + str(error), file=sys.stderr)
        print("Logs: " + str(folder.relative_to(ROOT)), file=sys.stderr)
        return 1
    finally:
        (folder / "summary.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
    for result in results:
        print("PASS: " + json.dumps(result, ensure_ascii=True))
    print("Logs: " + str(folder.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--pty-child":
        raise SystemExit(child_wrapper(sys.argv[2:]))
    raise SystemExit(main())
