#ifndef SIMTOS_PLC_CONNECTION_H
#define SIMTOS_PLC_CONNECTION_H

#include "Common.h"
#include <string>
#include <map>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>

namespace Simtos {

// PLC 주소 맵 (WOOSU_FRAMEWORK의 address_map.h 방식)
struct PlcAddressMap {
    std::map<std::string, std::string> writeAddresses;  // PC -> PLC
    std::map<std::string, std::string> readAddresses;   // PLC -> PC
};

class PlcConnection {
public:
    PlcConnection();
    ~PlcConnection();

    bool Connect(const std::string& ip, int port);
    void Disconnect();
    bool IsConnected() const;

    // 값 쓰기/읽기
    bool SetValue(const std::string& key, int value);
    int  GetValue(const std::string& key);

    // 서보 제어 패턴: 명령 비트 ON -> Ack 대기 -> 명령 비트 OFF
    bool SendAndWaitAck(const std::string& commandKey, const std::string& ackKey, int timeout_ms = 10000);

    // 주소 맵 로드
    bool LoadAddressMap(const std::string& configPath);

private:
    bool SendData(const std::string& data);
    std::string ReceiveData(int timeout_ms);

    SocketType socket_ = INVALID_SOCKET_VALUE;
    std::atomic<bool> connected_{false};
    std::mutex mutex_;
    PlcAddressMap addressMap_;
};

} // namespace Simtos

#endif // SIMTOS_PLC_CONNECTION_H
