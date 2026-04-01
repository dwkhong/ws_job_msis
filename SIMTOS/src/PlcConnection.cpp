#include "../include/PlcConnection.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
#endif

namespace Simtos {

PlcConnection::PlcConnection() {}

PlcConnection::~PlcConnection() {
    Disconnect();
}

bool PlcConnection::Connect(const std::string& ip, int port) {
    std::lock_guard<std::mutex> lock(mutex_);

    socket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_ == INVALID_SOCKET_VALUE) {
        std::cerr << "[PLC] Socket creation failed" << std::endl;
        return false;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, ip.c_str(), &addr.sin_addr);

    if (connect(socket_, (sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "[PLC] Connection failed to " << ip << ":" << port << std::endl;
#ifdef _WIN32
        closesocket(socket_);
#else
        close(socket_);
#endif
        socket_ = INVALID_SOCKET_VALUE;
        return false;
    }

    connected_ = true;
    std::cout << "[PLC] Connected to " << ip << ":" << port << std::endl;
    return true;
}

void PlcConnection::Disconnect() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (socket_ != INVALID_SOCKET_VALUE) {
#ifdef _WIN32
        closesocket(socket_);
#else
        close(socket_);
#endif
        socket_ = INVALID_SOCKET_VALUE;
        connected_ = false;
        std::cout << "[PLC] Disconnected" << std::endl;
    }
}

bool PlcConnection::IsConnected() const {
    return connected_.load();
}

bool PlcConnection::SetValue(const std::string& key, int value) {
    std::lock_guard<std::mutex> lock(mutex_);
    // TODO: 실제 PLC 프로토콜에 맞게 구현 (Modbus, MC Protocol 등)
    // 지금은 프레임워크만 잡아놓음
    std::string address = addressMap_.writeAddresses[key];
    std::string packet = address + "," + std::to_string(value) + "\n";
    return SendData(packet);
}

int PlcConnection::GetValue(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    // TODO: 실제 PLC 프로토콜에 맞게 구현
    std::string address = addressMap_.readAddresses[key];
    std::string packet = "READ," + address + "\n";
    if (!SendData(packet)) return -1;

    std::string response = ReceiveData(1000);
    if (response.empty()) return -1;

    try {
        return std::stoi(response);
    } catch (...) {
        return -1;
    }
}

bool PlcConnection::SendAndWaitAck(const std::string& commandKey, const std::string& ackKey, int timeout_ms) {
    // 1) 명령 비트 ON
    if (!SetValue(commandKey, 1)) {
        std::cerr << "[PLC] Failed to set " << commandKey << std::endl;
        return false;
    }

    // 2) Ack 대기
    auto start = std::chrono::steady_clock::now();
    while (true) {
        if (GetValue(ackKey) == 1) break;

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed > timeout_ms) {
            std::cerr << "[PLC] Ack timeout for " << ackKey << std::endl;
            SetValue(commandKey, 0);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // 3) 명령 비트 OFF
    SetValue(commandKey, 0);

    std::cout << "[PLC] " << commandKey << " -> Ack received" << std::endl;
    return true;
}

bool PlcConnection::LoadAddressMap(const std::string& configPath) {
    std::ifstream file(configPath);
    if (!file.is_open()) {
        std::cerr << "[PLC] Failed to open address map: " << configPath << std::endl;
        return false;
    }

    std::string line;
    std::string section;
    while (std::getline(file, line)) {
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
        if (line.empty() || line[0] == '#') continue;

        if (line[0] == '[') {
            section = line.substr(1, line.find(']') - 1);
            continue;
        }

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = line.substr(0, eq);
        std::string addr = line.substr(eq + 1);

        if (section == "Write") addressMap_.writeAddresses[key] = addr;
        else if (section == "Read") addressMap_.readAddresses[key] = addr;
    }

    std::cout << "[PLC] Loaded address map: "
              << addressMap_.writeAddresses.size() << " write, "
              << addressMap_.readAddresses.size() << " read" << std::endl;
    return true;
}

bool PlcConnection::SendData(const std::string& data) {
    if (socket_ == INVALID_SOCKET_VALUE) return false;
    int sent = send(socket_, data.c_str(), (int)data.size(), 0);
    return sent > 0;
}

std::string PlcConnection::ReceiveData(int timeout_ms) {
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

} // namespace Simtos
