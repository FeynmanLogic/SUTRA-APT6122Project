
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <iostream>
#include <string>

int main() {
    constexpr int PORT = 8080;

    // 1. Create a TCP socket
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);

    if (server_fd < 0) {
        perror("socket");
        return 1;
    }

    // Allow quick reuse of the port
    int opt = 1;
    setsockopt(
        server_fd,
        SOL_SOCKET,
        SO_REUSEADDR,
        &opt,
        sizeof(opt)
    );

    // 2. Configure the server address
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(PORT);

    // 3. Bind the socket to the port
    if (bind(
            server_fd,
            reinterpret_cast<sockaddr*>(&address),
            sizeof(address)
        ) < 0) {
        perror("bind");
        close(server_fd);
        return 1;
    }

    // 4. Listen for incoming connections
    if (listen(server_fd, 10) < 0) {
        perror("listen");
        close(server_fd);
        return 1;
    }

    std::cout << "SUTRA TCP Router listening on "
              << "127.0.0.1:" << PORT << std::endl;

    // 5. Process clients sequentially for now
    while (true) {
        int client_fd = accept(
            server_fd, nullptr, nullptr
        );

        if (client_fd < 0) {
            perror("accept");
            continue;
        }

        std::cout << "Client connected!" << std::endl;

        char buffer[4096];

        ssize_t bytes = recv(
            client_fd,
            buffer,
            sizeof(buffer),
            0
        );

        if (bytes > 0) {
            std::string message(buffer, bytes);

            std::cout << "Received: "
                      << message << std::endl;

            std::string response =
                "SUTRA received: " + message;

            // Send the entire response
            size_t sent = 0;

            while (sent < response.size()) {
                ssize_t n = send(
                    client_fd,
                    response.data() + sent,
                    response.size() - sent,
                    MSG_NOSIGNAL
                );

                if (n <= 0) {
                    perror("send");
                    break;
                }

                sent += static_cast<size_t>(n);
            }
        }

        close(client_fd);
    }

    close(server_fd);
    return 0;
}
