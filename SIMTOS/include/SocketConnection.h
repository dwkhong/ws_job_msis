#ifndef SIMTOS_SOCKET_CONNECTION_H
#define SIMTOS_SOCKET_CONNECTION_H

#include "Common.h"
#include <mutex>
#include <atomic>

namespace Simtos {

class SocketConnection {
public:
    SocketConnection(const DeviceInfo& info, const CommConfig& config);
    ~SocketConnection();

    bool Connect();
    void Disconnect();
    bool IsConnected() const;

    // 명령 전송 후 "ok" 응답 대기 -> 완료 신호("done") 대기
    CommandResponse SendCommand(const std::string& command);

    const DeviceInfo& GetInfo() const { return info_; }

private:
    bool SendData(const std::string& data);
    std::string ReceiveData(int timeout_ms);
    bool WaitForResponse(const std::string& expected, int timeout_ms);

    DeviceInfo info_;
    CommConfig config_;
    SocketType socket_ = INVALID_SOCKET_VALUE;
    std::mutex mutex_;

    static std::atomic<bool> winsockInitialized_;
    static std::mutex winsockMutex_;
    static bool InitWinsock();
    static void CleanupWinsock();
};

} // namespace Simtos

#endif // SIMTOS_SOCKET_CONNECTION_H
