#ifndef SIMTOS_COMMON_H
#define SIMTOS_COMMON_H

#include <string>
#include <vector>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
    typedef SOCKET SocketType;
    #define INVALID_SOCKET_VALUE INVALID_SOCKET
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    typedef int SocketType;
    #define INVALID_SOCKET_VALUE -1
#endif

namespace Simtos {

enum class ConnectionStatus {
    DISCONNECTED,
    CONNECTED,
    ERROR
};

enum class CommandResult {
    SUCCESS,
    FAILED,
    TIMEOUT
};

struct DeviceInfo {
    std::string name;
    std::string ip;
    int port = 0;
    ConnectionStatus status = ConnectionStatus::DISCONNECTED;
};

struct CommConfig {
    int connection_retry_delay_ms = 500;
    int ok_response_timeout_ms   = 1000;
    int completion_timeout_ms    = 60000;
};

struct CommandResponse {
    CommandResult result = CommandResult::FAILED;
    std::string message;
};

} // namespace Simtos

#endif // SIMTOS_COMMON_H
