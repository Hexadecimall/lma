#!/usr/bin/env python3
"""Model-free HTTP Files smoke test; isolated HOME and child servers only."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


KEY = "local-smoke-test-key"
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(base, method, path, body=None, content_type=None, authenticated=True):
    headers = {}
    if authenticated:
        headers["Authorization"] = f"Bearer {KEY}"
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(base + path, body, headers, method=method)
    try:
        response = HTTP.open(req, timeout=3)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, response.headers, response.read()


def value(result, status=200):
    actual, _, body = result
    assert actual == status, (actual, body)
    return json.loads(body)


@contextlib.contextmanager
def server(binary, home, profile):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update(HOME=str(home), USERPROFILE=str(home),
               OLLAMA_MODELS=str(home / "missing-ollama"))
    env.pop("LMA_MODELS", None)
    with tempfile.TemporaryFile() as log:
        child = subprocess.Popen(
            [str(binary), "s", "-a", profile, "-pt", str(port), "-ay", KEY],
            cwd=home, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
        )
        try:
            deadline = time.monotonic() + 10
            while True:
                if child.poll() is not None:
                    log.seek(0)
                    raise AssertionError(log.read().decode(errors="replace"))
                try:
                    health = value(request(base, "GET", "/health"))
                    assert health == {"status": "ok", "loaded": False}, health
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise AssertionError("isolated server did not become ready")
                    time.sleep(0.05)
            yield base
            assert value(request(base, "GET", "/health"))["loaded"] is False
        finally:
            # Only terminate the subprocess created above, never a user's server.
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=lambda p: Path(p).resolve(strict=True))
    args = parser.parse_args()
    if sys.platform not in {"linux", "darwin"}:
        parser.error("secure Files storage requires Linux/Android/macOS")
    with tempfile.TemporaryDirectory(prefix="lma-files-http-") as directory:
        home = Path(directory).resolve()
        store = home / ".lma/openai/files"
        payload = b"local uploaded bytes\x00\xff\r\n"
        body = (
            b'--lma-smoke\r\nContent-Disposition: form-data; name="purpose"\r\n\r\n'
            b'user_data\r\n--lma-smoke\r\nContent-Disposition: form-data; '
            b'name="file"; filename="sample.bin"\r\n'
            b'Content-Type: application/octet-stream\r\n\r\n'
            + payload + b'\r\n--lma-smoke--\r\n'
        )
        with server(args.binary, home, "openai") as base:
            error = value(request(base, "POST", "/v1/files", body,
                                  "multipart/form-data; boundary=lma-smoke",
                                  authenticated=False), 401)
            assert error["error"]["code"] == "invalid_api_key", error
            assert not store.exists()
            assert value(request(base, "GET", "/v1/files"))["data"] == []
            assert not store.exists(), "listing missing storage must not create it"
            uploaded = value(request(base, "POST", "/v1/files", body,
                                     "multipart/form-data; boundary=lma-smoke"))
            assert uploaded["bytes"] == len(payload), uploaded
            assert uploaded["status"] == "uploaded", uploaded
            file_id = uploaded["id"]
            path = f"/v1/files/{file_id}"
            assert value(request(base, "GET", path)) == uploaded
            status, headers, content = request(base, "GET", path + "/content")
            assert status == 200 and content == payload, (status, content)
            assert headers["Cache-Control"] == "no-store"
            status, headers, content = request(base, "HEAD", path + "/content")
            assert status == 200 and not content
            assert int(headers["Content-Length"]) == len(payload)
            listing = value(request(base, "GET", "/v1/files?limit=1&order=asc"))
            assert listing["data"] == [uploaded] and listing["has_more"] is False
            assert store.is_dir()
            assert store.stat().st_mode & 0o077 == 0
        with server(args.binary, home, "all") as base:
            assert value(request(base, "GET", path)) == uploaded
            assert request(base, "GET", path + "/content")[2] == payload
            deleted = value(request(base, "DELETE", path))
            assert deleted == {"id": file_id, "object": "file", "deleted": True}
            value(request(base, "GET", path), 404)
            assert value(request(base, "GET", "/v1/files"))["data"] == []
            assert not (store / (file_id + ".data")).exists()
            assert not (store / (file_id + ".json")).exists()
        print("PASS: authenticated Files upload/list/metadata/content/HEAD, "
              "restart persistence and deletion; no model loaded")


if __name__ == "__main__":
    main()
