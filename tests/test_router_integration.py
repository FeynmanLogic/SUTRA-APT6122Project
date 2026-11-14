"""Launch the router and verify the actual TCP -> scheduler -> worker path."""
import concurrent.futures
import json
import math
import os
import signal
import socket
import struct
import subprocess
import sys
import time


def recv_exact(sock, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Router closed the connection")
        data.extend(chunk)
    return bytes(data)


def envelope(prompt, **kwargs):
    return {"version": 1, "type": "infer", "payload": {"prompt": prompt, **kwargs}}


def main():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = subprocess.Popen(
        [sys.argv[1], "--port", str(port), "--backend", "mock"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    def exchange(message, fragmented=False):
        data = message if isinstance(message, bytes) else json.dumps(message).encode()
        frame = struct.pack("!I", len(data)) + data
        with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
            if fragmented:
                for byte in frame:
                    sock.sendall(bytes([byte]))
            else:
                sock.sendall(frame)
            length = struct.unpack("!I", recv_exact(sock, 4))[0]
            assert 0 < length <= 1024 * 1024
            return json.loads(recv_exact(sock, length))

    try:
        deadline = time.monotonic() + 15
        while True:
            if server.poll() is not None:
                raise RuntimeError("Router exited before accepting connections")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError("Router startup timed out")
                time.sleep(0.05)

        def request(i):
            prompt = f"client {i}: CUDA and Unicode \u03bb"
            result = exchange(envelope(prompt, max_new_tokens=8, priority=i % 3))
            assert result["type"] == "infer_result" and result["success"], result
            assert result["generated_text"] == "Synthetic response to: " + prompt, result
            assert result["worker_id"] == 0 and result["status"] == "completed", result
            for metric in ("queue_time_ms", "inference_time_ms", "latency_ms",
                           "time_to_first_token_ms", "tokens_per_second"):
                assert math.isfinite(result[metric]) and result[metric] >= 0, result
            return result["request_id"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            ids = list(pool.map(request, range(30)))
        assert len(set(ids)) == 30 and min(ids) > 0
        invalid = [
            envelope(""), envelope("x" * (16 * 1024 + 1)),
            envelope("x", max_new_tokens=0), envelope("x", max_new_tokens=513),
            envelope("x", max_new_tokens=2**32 + 64), envelope("x", max_new_tokens=True),
            envelope("x", priority=-1), envelope("x", priority=2**32),
            envelope("x", priority=0.5), {"version": 99, "type": "infer", "payload": {}},
            [], b"not JSON", {"version": 1, "type": "infer", "payload": {}},
        ]
        for message in invalid:
            assert exchange(message)["type"] == "error", message
        assert exchange(envelope("fragmented"), fragmented=True)["success"]
        for length in (0, 1024 * 1024 + 1):
            with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
                sock.sendall(struct.pack("!I", length))
                assert sock.recv(1) == b""
        # A client leaving mid-frame must not prevent the next request.
        with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
            sock.sendall(struct.pack("!I", 100) + b"{")
        request(100)
        print("PASS: concurrent routing, prompt/ID correlation, worker metrics, invalid/fragmented frames")
    finally:
        if server.poll() is None:
            server.send_signal(signal.SIGTERM)
        try:
            output, _ = server.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            output, _ = server.communicate()
            raise RuntimeError("Router did not stop within 15 seconds")
        if server.returncode not in (0, -signal.SIGTERM) and os.name != "nt":
            raise RuntimeError(f"Router failed during shutdown: {output}")


if __name__ == "__main__":
    main()
