#include "../include/SocketConnection.h"
#include <iostream>
#include <cstring>
#include <chrono>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <poll.h>
#endif

namespace Simtos {

std::atomic<bool> SocketConnection::winsockInitialized_{false};
std::mutex SocketConnection::winsockMutex_;

bool SocketConnection::InitWinsock() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(winsockMutex_);
    if (!winsockInitialized_) {
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            std::cerr << "[SocketConnection] WSAStartup failed" << std::endl;
            return false;
        }
        winsockInitialized_ = true;
    }
#endif
    return true;
}

void SocketConnection::CleanupWinsock() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(winsockMutex_);
    if (winsockInitialized_) {
        WSACleanup();
        winsockInitialized_ = false;
    }
#endif
}

SocketConnection::SocketConnection(const DeviceInfo& info, const CommConfig& config)
    : info_(info), config_(config) {}

SocketConnection::~SocketConnection() {
    Disconnect();
}

bool SocketConnection::Connect() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!InitWinsock()) return false;

    socket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_ == INVALID_SOCKET_VALUE) {
        std::cerr << "[" << info_.name << "] Socket creation failed" << std::endl;
        return false;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(info_.port);
    inet_pton(AF_INET, info_.ip.c_str(), &addr.sin_addr);

    if (connect(socket_, (sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "[" << info_.name << "] Connection failed to "
                  << info_.ip << ":" << info_.port << std::endl;
#ifdef _WIN32
        closesocket(socket_);
#else
        close(socket_);
#endif
        socket_ = INVALID_SOCKET_VALUE;
        return false;
    }

    info_.status = ConnectionStatus::CONNECTED;
    std::cout << "[" << info_.name << "] Connected to "
              << info_.ip << ":" << info_.port << std::endl;
    return true;
}

void SocketConnection::Disconnect() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (socket_ != INVALID_SOCKET_VALUE) {
#ifdef _WIN32
        closesocket(socket_);
#else
        close(socket_);
#endif
        socket_ = INVALID_SOCKET_VALUE;
        info_.status = ConnectionStatus::DISCONNECTED;
        std::cout << "[" << info_.name << "] Disconnected" << std::endl;
    }
}

bool SocketConnection::IsConnected() const {
    return info_.status == ConnectionStatus::CONNECTED;
}

CommandResponse SocketConnection::SendCommand(const std::string& command) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (socket_ == INVALID_SOCKET_VALUE) {
        return {CommandResult::FAILED, "Not connected"};
    }

    // 1) 명령 전송
    if (!SendData(command)) {
        return {CommandResult::FAILED, "Send failed"};
    }
    std::cout << "[" << info_.name << "] Sent: " << command << std::endl;

    // 2) "ok" 응답 대기
    if (!WaitForResponse("ok", config_.ok_response_timeout_ms)) {
        return {CommandResult::TIMEOUT, "No OK response"};
    }

    // 3) "done" 완료 신호 대기
    if (!WaitForResponse("done", config_.completion_timeout_ms)) {
        return {CommandResult::TIMEOUT, "No completion signal"};
    }

    std::cout << "[" << info_.name << "] Command " << command << " completed" << std::endl;
    return {CommandResult::SUCCESS, "OK"};
}

bool SocketConnection::SendData(const std::string& data) {
    int sent = send(socket_, data.c_str(), (int)data.size(), 0);
    return sent > 0;
}

std::string SocketConnection::ReceiveData(int timeout_ms) {
    // 타임아웃 설정
#ifdef _WIN32
    DWORD tv = timeout_ms;
    setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
#else
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif

    char buf[1024] = {0};
    int received = recv(socket_, buf, sizeof(buf) - 1, 0);
    if (received <= 0) return "";
    return std::string(buf, received);
}

bool SocketConnection::WaitForResponse(const std::string& expected, int timeout_ms) {
    std::string response = ReceiveData(timeout_ms);
    // 응답에 expected 문자열이 포함되어 있으면 OK
    return response.find(expected) != std::string::npos;
}

} // namespace Simtos
