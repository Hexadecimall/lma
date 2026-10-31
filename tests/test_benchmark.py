"""Model-free benchmark safety, timing, and HTTP-contract regression tests."""

import contextlib
import importlib.util
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "lma_benchmark", Path(__file__).with_name("benchmark.py")
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class BenchmarkTests(unittest.TestCase):
    def client(self, frames):
        client = bench.Client("http://127.0.0.1:1")
        data = b"".join((json.dumps(frame) + "\n").encode() for frame in frames)
        client.open = lambda *args: io.BytesIO(data)
        return client

    def test_ttft_measures_first_nonempty_text_not_lifecycle(self):
        client = self.client(
            [
                {"response": "", "done": False},
                {"response": "hello", "done": False},
                {"response": "!", "done": False},
                {"done": True, "eval_count": 2, "prompt_eval_count": 20},
            ]
        )
        times = iter([10, 10.1, 10.3, 10.4, 10.5])
        result = client.sample({}, clock=lambda: next(times))
        self.assertAlmostEqual(result["ttft_ms"], 300)
        self.assertAlmostEqual(result["elapsed_ms"], 500)
        self.assertEqual(result["end_to_end_tokens_per_second"], 4)
        self.assertEqual(result["reported_prompt_tokens"], 20)
        self.assertIsNone(result["server_prompt_eval_duration_ns"])
        self.assertIsNone(result["server_eval_duration_ns"])
        self.assertEqual(result["output_sha256"], bench.sha256("hello!"))
        self.assertIsNone(result["server_prompt_eval_tokens_per_second"])
        self.assertIsNone(result["server_eval_tokens_per_second"])

    def test_phase_metrics_only_when_server_supplies_real_durations(self):
        client = self.client(
            [
                {
                    "done": True,
                    "response": "x",
                    "eval_count": 2,
                    "prompt_eval_count": 10,
                    "eval_duration": 1_000_000_000,
                    "prompt_eval_duration": 2_000_000_000,
                }
            ]
        )
        times = iter([0, 3])
        result = client.sample({}, clock=lambda: next(times))
        self.assertEqual(result["server_eval_tokens_per_second"], 2)
        self.assertEqual(result["server_prompt_eval_tokens_per_second"], 5)
        self.assertAlmostEqual(result["end_to_end_tokens_per_second"], 2 / 3)

    def test_empty_output_has_no_invented_first_token(self):
        result = self.client(
            [
                {
                    "done": True,
                    "eval_count": 1,
                    "prompt_eval_count": 3,
                    "eval_duration": 0,
                }
            ]
        ).sample({})
        self.assertIsNone(result["ttft_ms"])
        self.assertIsNone(result["server_eval_tokens_per_second"])

    def test_returned_model_identity_is_checked_on_every_frame_when_supplied(self):
        terminal = {"done": True, "eval_count": 1, "prompt_eval_count": 3}
        payload = {"model": "model:latest"}
        for identity in ("other:latest", "model", "", None, 1, True, ["model:latest"]):
            for terminal_frame in (False, True):
                frames = (
                    [dict(terminal, model=identity)]
                    if terminal_frame
                    else [{"model": identity, "response": "x"}, terminal]
                )
                with self.subTest(identity=identity, terminal=terminal_frame):
                    with self.assertRaisesRegex(bench.BenchmarkError, "model identity"):
                        self.client(frames).sample(payload)

    def test_matching_or_omitted_response_identity_is_accepted(self):
        terminal = {"done": True, "eval_count": 1, "prompt_eval_count": 3}
        for frames in (
            [
                {"model": "model:latest", "response": "x"},
                dict(terminal, model="model:latest"),
            ],
            [{"response": "x"}, terminal],
        ):
            with self.subTest(frames=frames):
                result = self.client(frames).sample({"model": "model:latest"})
                self.assertEqual(result["output_sha256"], bench.sha256("x"))

    def test_invalid_and_unfinished_streams_rejected(self):
        for frames in (
            [{"response": "x"}],
            [{"error": "private server error"}],
            [{"response": 1}],
            [{"thinking": "secret"}],
            [1],
            [{"done": True, "eval_count": True, "prompt_eval_count": 3}],
            [
                {
                    "done": True,
                    "eval_count": 1,
                    "prompt_eval_count": 3,
                    "eval_duration": -1,
                }
            ],
        ):
            with self.subTest(frames=frames), self.assertRaises(bench.BenchmarkError):
                self.client(frames).sample({})
        client = bench.Client("http://localhost:1")
        client.open = lambda *args: io.BytesIO(b"not json\n")
        with self.assertRaises(bench.BenchmarkError):
            client.sample({})

    def test_generation_byte_limits_and_time_budget(self):
        client = bench.Client("http://localhost:1")
        client.open = lambda *args: io.BytesIO(b"x" * (bench.MAX_LINE + 1))
        with self.assertRaises(bench.BenchmarkError):
            client.sample({})
        client = self.client([{"response": "x"}])
        times = iter([0, 121])
        with self.assertRaises(bench.BenchmarkError):
            client.sample({}, clock=lambda: next(times))

    def test_preflight_does_not_load_unload_or_download(self):
        client = bench.Client("http://localhost:1")
        calls = []

        def get(path):
            calls.append(path)
            return {"models": []}

        client.get = get
        with self.assertRaises(bench.BenchmarkError):
            client.preflight("model:latest", 4096)
        self.assertEqual(calls, ["/api/ps"])
        with self.assertRaises(bench.BenchmarkError):
            client.preflight("model:latest", 4096, allow_load=True)
        self.assertEqual(calls, ["/api/ps", "/api/ps", "/api/tags"])
        client.get = lambda path: {
            "models": [{"model": "model:latest", "context_length": 4096}]
        }
        self.assertTrue(client.preflight("model:latest", 4096)["was_resident"])
        with self.assertRaises(bench.BenchmarkError):
            client.preflight(
                "model", 4096
            )  # Never guess an untagged/ambiguous identity.

    def test_resident_context_must_be_known_positive_integer_and_match(self):
        client = bench.Client("http://localhost:1")
        for metadata in (
            {},
            {"context_length": None},
            {"context_length": "4096"},
            {"context_length": True},
            {"context_length": 4096.0},
            {"context_length": 0},
            {"context_length": -1},
            {"context_length": 2048},
            {"context_length": 8192},
        ):
            with self.subTest(metadata=metadata):
                resident = dict(metadata, name="model:latest")
                with patch.object(
                    client, "get", return_value={"models": [resident]}
                ) as get:
                    with self.assertRaisesRegex(bench.BenchmarkError, "context_length"):
                        client.preflight("model:latest", 4096)
                    get.assert_called_once_with("/api/ps")
        for name_field in ("name", "model"):
            resident = {name_field: "model:latest", "context_length": 4096}
            with self.subTest(name_field=name_field):
                with patch.object(
                    client, "get", return_value={"models": [resident]}
                ) as get:
                    self.assertEqual(
                        client.preflight("model:latest", 4096),
                        {"was_resident": True, "model": resident},
                    )
                    get.assert_called_once_with("/api/ps")

    def test_allow_load_explicitly_permits_unknown_or_different_resident_context(self):
        client = bench.Client("http://localhost:1")
        for metadata in ({}, {"context_length": None}, {"context_length": 2048}):
            resident = dict(metadata, name="model:latest")
            with self.subTest(metadata=metadata):
                with patch.object(
                    client, "get", return_value={"models": [resident]}
                ) as get:
                    result = client.preflight("model:latest", 4096, allow_load=True)
                    self.assertTrue(result["was_resident"])
                    get.assert_called_once_with("/api/ps")

    def test_allow_load_checks_exact_installed_name_without_downloading(self):
        client = bench.Client("http://localhost:1")
        installed = {"name": "model:latest"}
        with patch.object(
            client, "get", side_effect=[{"models": []}, {"models": [installed]}]
        ) as get:
            self.assertEqual(
                client.preflight("model:latest", 4096, allow_load=True),
                {"was_resident": False, "model": installed},
            )
            self.assertEqual(
                [call.args[0] for call in get.call_args_list], ["/api/ps", "/api/tags"]
            )
        with patch.object(
            client, "get", side_effect=[{"models": []}, {"models": [installed]}]
        ):
            with self.assertRaisesRegex(bench.BenchmarkError, "exact model name"):
                client.preflight("model", 4096, allow_load=True)

    def test_run_rejects_unsafe_context_before_any_generation_or_report(self):
        with tempfile.TemporaryDirectory() as folder:
            prompt = Path(folder) / "prompt.txt"
            prompt.write_text("test prompt", encoding="utf-8")
            output = Path(folder) / "result.json"
            args = [
                "--base-url",
                "http://localhost:1",
                "--model",
                "model:latest",
                "--label",
                "fixture",
                "--prompt-file",
                str(prompt),
                "--run",
                "--context",
                "4096",
                "--output",
                str(output),
            ]
            for metadata in ({}, {"context_length": None}, {"context_length": 8192}):
                inventory = {"models": [dict(metadata, name="model:latest")]}
                with self.subTest(metadata=metadata):
                    with patch.object(
                        bench.Client,
                        "open",
                        return_value=io.BytesIO(json.dumps(inventory).encode()),
                    ) as open_request:
                        with (
                            contextlib.redirect_stderr(io.StringIO()),
                            self.assertRaises(SystemExit) as error,
                        ):
                            bench.main(args)
                        self.assertEqual(error.exception.code, 2)
                        open_request.assert_called_once_with("/api/ps")
                    self.assertFalse(output.exists())

    def test_remote_prompt_sharing_requires_explicit_opt_in(self):
        self.assertEqual(
            bench.base_url("http://127.0.0.1:1234/"), "http://127.0.0.1:1234"
        )
        self.assertEqual(bench.base_url("http://[::1]:1234"), "http://[::1]:1234")
        for url in (
            "https://example.com",
            "http://secret@localhost",
            "file:///tmp/model",
            "http://localhost?key=secret",
            "http://localhost/#fragment",
            "http://localhost:bad",
        ):
            with self.subTest(url=url), self.assertRaises(bench.BenchmarkError):
                bench.base_url(url)
        self.assertEqual(
            bench.base_url("https://example.com", True), "https://example.com"
        )
        with self.assertRaises(bench.BenchmarkError):
            bench.NoRedirect().redirect_request(
                None, None, 302, "", {}, "https://example.com"
            )

    def test_plan_is_network_free_does_not_copy_prompt_and_never_overwrites_results(
        self,
    ):
        with tempfile.TemporaryDirectory() as folder:
            prompt = Path(folder) / "prompt.txt"
            prompt.write_text("private benchmark prompt", encoding="utf-8")
            output = Path(folder) / "result.json"
            args = [
                "--base-url",
                "http://127.0.0.1:8080",
                "--model",
                "model:latest",
                "--label",
                "test",
                "--prompt-file",
                str(prompt),
                "--output",
                str(output),
            ]
            with patch.object(bench, "Client", side_effect=AssertionError("network")):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(bench.main(args), 0)
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured):
                    self.assertEqual(bench.main(args[:-2] + ["--allow-load"]), 0)
                self.assertFalse(json.loads(captured.getvalue())["run"])
                self.assertTrue(json.loads(captured.getvalue())["allow_load"])
            report = json.loads(output.read_text())
            self.assertFalse(report["run"])
            self.assertEqual(report["samples"], [])
            self.assertEqual(report["measured_requests"], 9)
            self.assertEqual(report["warmup_requests"], 9)
            self.assertNotIn("private benchmark prompt", output.read_text())
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                bench.main(args)
            self.assertEqual(json.loads(output.read_text()), report)

    def test_mismatched_priming_identity_aborts_before_measurement_or_report(self):
        inventory = {"models": [{"name": "model:latest", "context_length": 4096}]}
        frames = b'{"model":"other:latest","done":true,"eval_count":1,"prompt_eval_count":3}\n'
        with tempfile.TemporaryDirectory() as folder:
            prompt = Path(folder) / "prompt.txt"
            prompt.write_text("test prompt", encoding="utf-8")
            output = Path(folder) / "result.json"
            args = [
                "--base-url",
                "http://localhost:1",
                "--model",
                "model:latest",
                "--label",
                "fixture",
                "--prompt-file",
                str(prompt),
                "--run",
                "--output",
                str(output),
            ]
            with patch.object(
                bench.Client,
                "open",
                side_effect=[
                    io.BytesIO(json.dumps(inventory).encode()),
                    io.BytesIO(frames),
                ],
            ) as open_request:
                errors = io.StringIO()
                with (
                    contextlib.redirect_stderr(errors),
                    self.assertRaises(SystemExit) as error,
                ):
                    bench.main(args)
                self.assertEqual(error.exception.code, 2)
                self.assertIn("model identity", errors.getvalue())
                self.assertEqual(
                    [call.args[0] for call in open_request.call_args_list],
                    ["/api/ps", "/api/generate"],
                )
                self.assertEqual(
                    open_request.call_args_list[1].args[1]["prompt"],
                    bench.prompts("test prompt")[0][1],
                )
            self.assertFalse(output.exists())

    def test_full_http_run_is_sequential_raw_and_uses_only_resident_model(self):
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, payload):
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                calls.append((self.path, None))
                self.send(
                    json.dumps(
                        {
                            "models": [
                                {
                                    "name": "model:latest",
                                    "digest": "fixture",
                                    "context_length": 4096,
                                }
                            ]
                        }
                    ).encode()
                )

            def do_POST(self):
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                calls.append((self.path, payload))
                self.send(
                    b'{"model":"model:latest","response":"ok","done":false}\n'
                    b'{"model":"model:latest","done":true,"eval_count":2,"prompt_eval_count":3}\n'
                )

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                prompt = Path(folder) / "prompt.txt"
                prompt.write_text("test prompt", encoding="utf-8")
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured):
                    bench.main(
                        [
                            "--base-url",
                            f"http://127.0.0.1:{server.server_port}",
                            "--model",
                            "model:latest",
                            "--label",
                            "fixture",
                            "--run",
                            "--runs",
                            "2",
                            "--prompt-file",
                            str(prompt),
                        ]
                    )
                report = json.loads(captured.getvalue())
            self.assertEqual(calls[0], ("/api/ps", None))
            self.assertEqual(len(calls), 13)  # 6 priming + 6 measured, one preflight.
            base, extended, changed = [text for _, text in bench.prompts("test prompt")]
            self.assertEqual(
                [request["prompt"] for _, request in calls[1:]],
                [base, base, base, extended, base, changed] * 2,
            )
            self.assertEqual(report["warmup_requests"], 6)
            self.assertEqual(report["measured_requests"], 6)
            for path, request in calls[1:]:
                self.assertEqual(path, "/api/generate")
                self.assertTrue(request["raw"])
                self.assertTrue(request["stream"])
                self.assertEqual(request["model"], "model:latest")
                self.assertNotIn("keep_alive", request)
                self.assertEqual(request["options"]["temperature"], 0)
            self.assertEqual(len(report["samples"]), 6)
            self.assertEqual(
                [
                    (sample["workload"], sample["repetition"])
                    for sample in report["samples"]
                ],
                [
                    (name, repetition)
                    for repetition in (1, 2)
                    for name in ("identical", "extended", "changed_suffix")
                ],
            )
            self.assertEqual(
                [sample["prompt_sha256"] for sample in report["samples"]],
                [bench.sha256(text) for text in (base, extended, changed)] * 2,
            )
            self.assertEqual(report["summary"]["identical"]["samples"], 2)
            self.assertEqual(report["summary"]["identical"]["distinct_outputs"], 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
