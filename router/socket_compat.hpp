#pragma once

#include <cstddef>
#include <cstdint>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
using Socket = SOCKET;
constexpr Socket invalid_socket = INVALID_SOCKET;
inline void closeSocket(Socket fd) { closesocket(fd); }
inline bool socketInterrupted() { return WSAGetLastError() == WSAEINTR; }
class SocketRuntime {
public:
    SocketRuntime() { WSADATA data{}; ready = WSAStartup(MAKEWORD(2, 2), &data) == 0; }
    ~SocketRuntime() { if (ready) WSACleanup(); }
    bool ready;
};
inline void setSocketTimeout(Socket fd, int option, int seconds) {
    DWORD timeout = static_cast<DWORD>(seconds * 1000);
    setsockopt(fd, SOL_SOCKET, option, reinterpret_cast<const char*>(&timeout), sizeof(timeout));
}
#else
#include <arpa/inet.h>
#include <cerrno>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
using Socket = int;
constexpr Socket invalid_socket = -1;
inline void closeSocket(Socket fd) { close(fd); }
inline bool socketInterrupted() { return errno == EINTR; }
struct SocketRuntime { bool ready{true}; };
inline void setSocketTimeout(Socket fd, int option, int seconds) {
    timeval timeout{};
    timeout.tv_sec = seconds;
    setsockopt(fd, SOL_SOCKET, option, &timeout, sizeof(timeout));
}
#endif

inline int receiveBytes(Socket fd, char* buffer, std::size_t size) {
    return static_cast<int>(recv(fd, buffer, static_cast<int>(size), 0));
}
inline int sendBytes(Socket fd, const char* buffer, std::size_t size) {
#ifdef MSG_NOSIGNAL
    constexpr int flags = MSG_NOSIGNAL;
#else
    constexpr int flags = 0;
#endif
    return static_cast<int>(send(fd, buffer, static_cast<int>(size), flags));
}
