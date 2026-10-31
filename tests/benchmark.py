#!/usr/bin/env python3
"""Explicit, sequential raw-text benchmark for lma/Ollama HTTP servers.

Default is a network-free plan. --run requires the model already resident at
the requested context; --allow-load separately permits loading/reconfiguring an
installed model. Never starts a process, downloads weights, unloads a model,
or sends keep_alive overrides.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

MAX_BYTES = 8 * 1024 * 1024
MAX_LINE = 1024 * 1024


class BenchmarkError(Exception):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise BenchmarkError(
            "HTTP redirects are disabled; use the final server URL explicitly"
        )


def base_url(value, allow_remote=False):
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise BenchmarkError(
            "Use an HTTP(S) base URL without credentials, query, or fragment"
        )
    if not allow_remote and parsed.hostname.lower() not in (
        "localhost",
        "127.0.0.1",
        "::1",
    ):
        raise BenchmarkError(
            "Remote servers receive prompt text; opt in with --allow-remote"
        )
    try:
        parsed.port
    except ValueError as error:
        raise BenchmarkError("Invalid server port") from error
    return value.rstrip("/")


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompts(text):
    if not text.strip() or len(text.encode("utf-8")) > 1024 * 1024 - 256:
        raise BenchmarkError("Prompt must be nonempty and smaller than 1 MiB")
    base = text + "\n\nAnswer:"
    return [
        ("identical", base),
        ("extended", base + "\nContinue with more detail:"),
        ("changed_suffix", text + "\n\nSummarize:"),
    ]


def integer(value, field):
    if type(value) is not int or value < 0:
        raise BenchmarkError(f"{field} must be a nonnegative integer")
    return value


def rate(count, seconds):
    return count / seconds if seconds and seconds > 0 else None


class Client:
    def __init__(self, url, timeout=120, api_key=None):
        self.url = url
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = "Bearer " + api_key
        # Ignore proxy environment for local endpoints. Redirects are never followed.
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirect()
        )

    def open(self, path, payload=None):
        request = urllib.request.Request(
            self.url + path,
            data=None
            if payload is None
            else json.dumps(payload, allow_nan=False).encode(),
            headers=self.headers,
            method="GET" if payload is None else "POST",
        )
        try:
            return self.opener.open(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            raise BenchmarkError(
                f"{path}: HTTP {error.code}; check the server log"
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise BenchmarkError(
                f"{path}: server connection failed ({type(error).__name__})"
            ) from error

    def get(self, path):
        with self.open(path) as response:
            body = response.read(MAX_BYTES + 1)
        if len(body) > MAX_BYTES:
            raise BenchmarkError("Server metadata exceeds 8 MiB")
        try:
            result = json.loads(body)
        except (ValueError, UnicodeError) as error:
            raise BenchmarkError("Invalid JSON server metadata") from error
        if not isinstance(result, dict):
            raise BenchmarkError("Expected JSON object server metadata")
        return result

    def preflight(self, model, context, allow_load=False):
        inventory = self.get("/api/ps")
        resident = find_model(inventory, model)
        if resident is not None:
            if not allow_load:
                resident_context = resident.get("context_length")
                if type(resident_context) is not int or resident_context <= 0:
                    raise BenchmarkError(
                        "Resident context_length is unknown or invalid; verify the server's context "
                        "before running, or explicitly use --allow-load"
                    )
                if resident_context != context:
                    raise BenchmarkError(
                        f"Resident context_length {resident_context} does not match requested context {context}; "
                        "configure it yourself, or explicitly use --allow-load"
                    )
            return {"was_resident": True, "model": resident}
        if not allow_load:
            raise BenchmarkError(
                "Requested model is not resident; load it yourself, or explicitly use --allow-load"
            )
        installed = find_model(self.get("/api/tags"), model)
        if installed is None:
            raise BenchmarkError(
                "Requested exact model name is not installed; nothing was generated or downloaded"
            )
        return {"was_resident": False, "model": installed}

    def sample(self, payload, clock=time.perf_counter):
        started = clock()
        first_text = None
        output = hashlib.sha256()
        output_bytes = 0
        received = 0
        terminal = None
        with self.open("/api/generate", payload) as response:
            while True:
                line = response.readline(MAX_LINE + 1)
                now = clock()
                if now - started > self.timeout:
                    raise BenchmarkError("Generation exceeded the request time budget")
                if not line:
                    break
                received += len(line)
                if len(line) > MAX_LINE or received > MAX_BYTES:
                    raise BenchmarkError("Generation stream exceeds its byte limit")
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except (ValueError, UnicodeError) as error:
                    raise BenchmarkError("Invalid NDJSON generation stream") from error
                if not isinstance(event, dict):
                    raise BenchmarkError("Expected object generation frame")
                if "error" in event:
                    raise BenchmarkError(
                        "Server reported generation failure; check its log"
                    )
                if "model" in event and event["model"] != payload["model"]:
                    raise BenchmarkError(
                        "Returned model identity does not match the requested exact model name"
                    )
                if event.get("thinking"):
                    raise BenchmarkError(
                        "Separate reasoning output is not comparable to this raw-text workload"
                    )
                text = event.get("response", "")
                if not isinstance(text, str):
                    raise BenchmarkError("Generation response must be text")
                if text:
                    if first_text is None:
                        first_text = now
                    encoded = text.encode("utf-8")
                    output.update(encoded)
                    output_bytes += len(encoded)
                if event.get("done") is True:
                    terminal = event
                    finished = now
                    break
        if terminal is None:
            raise BenchmarkError("Generation stream ended without done:true")
        generated = integer(terminal.get("eval_count"), "eval_count")
        prompt_count = integer(terminal.get("prompt_eval_count"), "prompt_eval_count")
        elapsed = finished - started
        result = {
            "elapsed_ms": elapsed * 1000,
            "ttft_ms": None if first_text is None else (first_text - started) * 1000,
            "output_tokens": generated,
            "reported_prompt_tokens": prompt_count,
            "end_to_end_tokens_per_second": rate(generated, elapsed),
            "output_bytes": output_bytes,
            "output_sha256": output.hexdigest(),
            "done_reason": terminal.get("done_reason"),
        }
        # These are reported count/duration ratios, not verified evaluated-token rates.
        # lma reports logical prompt usage and currently supplies no phase durations.
        for name, count in (("prompt_eval", prompt_count), ("eval", generated)):
            duration = terminal.get(name + "_duration")
            if duration is not None:
                integer(duration, name + "_duration")
            result["server_" + name + "_duration_ns"] = duration
            result["server_" + name + "_tokens_per_second"] = rate(
                count, None if duration is None else duration / 1e9
            )
        return result


def find_model(inventory, name):
    models = inventory.get("models")
    if not isinstance(models, list) or any(
        not isinstance(model, dict) for model in models
    ):
        raise BenchmarkError("Invalid model inventory")
    return next(
        (model for model in models if name in (model.get("name"), model.get("model"))),
        None,
    )


def summarize(samples):
    summary = {}
    for workload in ("identical", "extended", "changed_suffix"):
        group = [sample for sample in samples if sample["workload"] == workload]
        if not group:
            continue
        entry = {
            "samples": len(group),
            "distinct_outputs": len({sample["output_sha256"] for sample in group}),
        }
        for field in (
            "ttft_ms",
            "elapsed_ms",
            "end_to_end_tokens_per_second",
            "server_prompt_eval_tokens_per_second",
            "server_eval_tokens_per_second",
        ):
            values = [
                sample[field] for sample in group if sample.get(field) is not None
            ]
            entry["median_" + field] = statistics.median(values) if values else None
        summary[workload] = entry
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--model", required=True, help="Exact installed/resident name, including tag"
    )
    parser.add_argument(
        "--label", required=True, help="Engine/build label recorded in the report"
    )
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="New JSON file; existing results are never overwritten",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--api-key-env", help="Environment variable holding the API key; never saved"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Make real generation requests, one at a time",
    )
    parser.add_argument(
        "--allow-load",
        action="store_true",
        help="Permit loading/reconfiguring an already installed model",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Permit sending the prompt to a non-loopback server",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Record weights hash, build, GPU/batch/flash/KV settings and hardware",
    )
    args = parser.parse_args(argv)
    try:
        if (
            not 1 <= args.runs <= 100
            or not 1 <= args.max_tokens < args.context <= 4294967295
        ):
            raise BenchmarkError(
                "Use 1..100 runs and 1 <= max-tokens < context <= u32::MAX"
            )
        if not math.isfinite(args.timeout) or not 1 <= args.timeout <= 3600:
            raise BenchmarkError("Timeout must be 1..3600 seconds")
        if not args.model.strip() or not args.label.strip():
            raise BenchmarkError("Model and label must be nonempty")
        if args.output and args.output.exists():
            raise BenchmarkError("Output already exists; choose a new file")
        url = base_url(args.base_url, args.allow_remote)
        with args.prompt_file.open("rb") as source:
            encoded = source.read(1024 * 1024 + 1)
        if len(encoded) > 1024 * 1024:
            raise BenchmarkError("Prompt file exceeds 1 MiB")
        workloads = prompts(encoded.decode("utf-8"))
        options = {
            "num_predict": args.max_tokens,
            "num_ctx": args.context,
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
        }
        report = {
            "schema_version": 1,
            "label": args.label,
            "base_url": url,
            "model": args.model,
            "notes": args.notes,
            "options": options,
            "raw": True,
            "stream": True,
            "run": args.run,
            "allow_load": args.allow_load,
            "warmup_requests": args.runs * len(workloads),
            "measured_requests": args.runs * len(workloads),
            "workloads": [
                {"name": name, "prompt_sha256": sha256(text)}
                for name, text in workloads
            ],
            "samples": [],
        }
        if args.run:
            key = os.environ.get(args.api_key_env) if args.api_key_env else None
            if args.api_key_env and not key:
                raise BenchmarkError(
                    "The requested API-key environment variable is empty"
                )
            client = Client(url, args.timeout, key)
            report["preflight"] = client.preflight(
                args.model, args.context, args.allow_load
            )
            for repetition in range(args.runs):

                def payload(text):
                    return {
                        "model": args.model,
                        "prompt": text,
                        "raw": True,
                        "stream": True,
                        "options": options,
                    }

                for name, text in workloads:
                    # Independently measure base -> target, not the previous target
                    # -> target. This primes a warm prefix; it does not flush caches.
                    client.sample(payload(workloads[0][1]))
                    result = client.sample(payload(text))
                    result.update(
                        workload=name,
                        repetition=repetition + 1,
                        prompt_sha256=sha256(text),
                    )
                    report["samples"].append(result)
            report["summary"] = summarize(report["samples"])
        rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(rendered)
        print(rendered, end="")
        return 0
    except (BenchmarkError, OSError, UnicodeError, ValueError) as error:
        parser.exit(2, f"benchmark: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
