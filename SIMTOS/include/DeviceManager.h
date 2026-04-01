#ifndef SIMTOS_DEVICE_MANAGER_H
#define SIMTOS_DEVICE_MANAGER_H

#include "Common.h"
#include "SocketConnection.h"
#include <map>
#include <memory>
#include <string>

namespace Simtos {

class DeviceManager {
public:
    DeviceManager();
    ~DeviceManager();

    // config 파일에서 디바이스 정보 로드
    bool LoadConfig(const std::string& robotConfigPath, const std::string& commConfigPath);

    bool ConnectAll();
    void DisconnectAll();
    bool ConnectDevice(const std::string& name);
    bool IsConnected(const std::string& name) const;

    // 특정 디바이스에 명령 전송
    CommandResponse SendCommand(const std::string& deviceName, const std::string& command);

    // 디바이스가 완료 신호를 보낼 때까지 대기 (블로킹)
    CommandResponse WaitForCompletion(const std::string& deviceName);

    std::vector<std::string> GetDeviceNames() const;

private:
    std::shared_ptr<SocketConnection> FindDevice(const std::string& name) const;
    bool ParseRobotsIni(const std::string& path);
    bool ParseCommIni(const std::string& path);

    std::map<std::string, std::shared_ptr<SocketConnection>> devices_;
    CommConfig commConfig_;
};

} // namespace Simtos

#endif // SIMTOS_DEVICE_MANAGER_H
