
import socket
import struct
import json

HOST = "127.0.0.1"
PORT = 8080

cases = [
    ("Empty prompt", {"version": 1, "type": "infer",
                      "payload": {"prompt": ""}}),
    ("Too many tokens", {"version": 1, "type": "infer",
                         "payload": {"prompt": "Hello",
                                     "max_new_tokens": 1000}}),
    ("Invalid priority", {"version": 1, "type": "infer",
                          "payload": {"prompt": "Hello",
                                      "priority": 5}}),
    ("Wrong version", {"version": 99, "type": "infer",
                       "payload": {"prompt": "Hello"}}),
    ("Missing prompt", {"version": 1, "type": "infer",
                        "payload": {}})
]

def recv_exact(sock, n):
    result = b""
    while len(result) < n:
        chunk = sock.recv(n - len(result))
        if not chunk:
            raise ConnectionError("Connection closed")
        result += chunk
    return result

for name, message in cases:
    data = json.dumps(message).encode()

    with socket.create_connection((HOST, PORT), timeout=5) as sock:
        sock.settimeout(5)
        sock.sendall(struct.pack("!I", len(data)) + data)

        length = struct.unpack("!I", recv_exact(sock, 4))[0]
        response = json.loads(recv_exact(sock, length))

        assert response["type"] == "error", (name, response)
        print(f"PASS: {name} -> {response['error']}")

print("\nAll invalid request tests passed.")
