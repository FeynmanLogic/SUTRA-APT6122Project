
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <iostream>
#include <string>
#include <thread>

constexpr int PORT = 8080;
constexpr int MAX_CLIENTS = 32;

// Number of currently connected clients
std::atomic<int> active_clients{0};

void handleClient(int client_fd) {
    active_clients++;

    std::cout << "Client connected. Active clients: "
              << active_clients.load() << std::endl;

    char buffer[4096];

    while (true) {
        ssize_t bytes = recv(
            client_fd,
            buffer,
            sizeof(buffer),
            0
        );

        if (bytes <= 0) {
            if (bytes < 0 && errno == EINTR)
                continue;
            break;
        }

        std::string message(buffer, bytes);

        std::cout << "Received: "
                  << message << std::endl;

        std::string response =
            "SUTRA received: " + message;

        size_t sent = 0;

        while (sent < response.size()) {
            ssize_t n = send(
                client_fd,
                response.data() + sent,
                response.size() - sent,
                MSG_NOSIGNAL
            );

            if (n < 0 && errno == EINTR)
                continue;

            if (n <= 0)
                break;

            sent += static_cast<size_t>(n);
        }

        if (sent != response.size())
            break;
    }

    close(client_fd);
    active_clients--;

    std::cout << "Client disconnected. Active: "
              << active_clients.load() << std::endl;
}

int main() {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);

    if (server_fd < 0) {
        perror("socket");
        return 1;
    }

    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET,
                   SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt");
        close(server_fd);
        return 1;
    }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(PORT);

    if (bind(server_fd,
             reinterpret_cast<sockaddr*>(&address),
             sizeof(address)) < 0) {
        perror("bind");
        close(server_fd);
        return 1;
    }

    if (listen(server_fd, MAX_CLIENTS) < 0) {
        perror("listen");
        close(server_fd);
        return 1;
    }

    std::cout << "SUTRA multithreaded TCP Router "
              << "listening on 127.0.0.1:"
              << PORT << std::endl;

    while (true) {
        int client_fd = accept(
            server_fd, nullptr, nullptr
        );

        if (client_fd < 0) {
            if (errno == EINTR)
                continue;
            perror("accept");
            continue;
        }

        if (active_clients.load() >= MAX_CLIENTS) {
            close(client_fd);
            continue;
        }

        // Separate thread for each client
        std::thread(
            handleClient,
            client_fd
        ).detach();
    }

    close(server_fd);
    return 0;
}
