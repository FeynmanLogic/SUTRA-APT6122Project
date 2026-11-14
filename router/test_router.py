
import socket
import json
import struct
from concurrent.futures import ThreadPoolExecutor

HOST = "127.0.0.1"
PORT = 8080


def recv_exact(sock, size):
    data = bytearray()

    while len(data) < size:
        chunk = sock.recv(size - len(data))

        if not chunk:
            raise ConnectionError("Server disconnected")

        data.extend(chunk)

    return bytes(data)


def send_request(i):
    request = {
        "version": 1,
        "type": "infer",
        "payload": {
            "prompt": f"Explain CUDA for client {i}",
            "max_new_tokens": 64,
            "priority": i % 3
        }
    }

    payload = json.dumps(request).encode("utf-8")
    frame = struct.pack("!I", len(payload)) + payload

    with socket.create_connection(
        (HOST, PORT), timeout=10
    ) as sock:

        sock.settimeout(15)
        sock.sendall(frame)

        length_data = recv_exact(sock, 4)
        length = struct.unpack("!I", length_data)[0]

        response_data = recv_exact(sock, length)

        return json.loads(response_data)


with ThreadPoolExecutor(max_workers=10) as pool:
    results = list(pool.map(send_request, range(10)))


for response in results:
    print(
        f"Request {response['request_id']}: "
        f"{response['status']} | "
        f"{response['generated_text']}"
    )


assert len(results) == 10
assert all(r["success"] for r in results)
assert len({r["request_id"] for r in results}) == 10

print("\nPASS: All 10 requests completed with unique IDs.")
