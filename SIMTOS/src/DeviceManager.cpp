#include "../include/DeviceManager.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace Simtos {

DeviceManager::DeviceManager() {}
DeviceManager::~DeviceManager() { DisconnectAll(); }

bool DeviceManager::LoadConfig(const std::string& robotConfigPath, const std::string& commConfigPath) {
    if (!ParseCommIni(commConfigPath)) {
        std::cerr << "[DeviceManager] Failed to load comm config" << std::endl;
        return false;
    }
    if (!ParseRobotsIni(robotConfigPath)) {
        std::cerr << "[DeviceManager] Failed to load robot config" << std::endl;
        return false;
    }
    return true;
}

bool DeviceManager::ParseRobotsIni(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return false;

    DeviceInfo current;
    std::string line;
    while (std::getline(file, line)) {
        // 공백 제거
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
        if (line.empty() || line[0] == '#') continue;

        if (line[0] == '[') {
            // 이전 섹션 저장
            if (!current.name.empty()) {
                devices_[current.name] = std::make_shared<SocketConnection>(current, commConfig_);
            }
            current = DeviceInfo();
            continue;
        }

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);

        if (key == "name") current.name = val;
        else if (key == "ip") current.ip = val;
        else if (key == "port") current.port = std::stoi(val);
    }

    // 마지막 섹션 저장
    if (!current.name.empty()) {
        devices_[current.name] = std::make_shared<SocketConnection>(current, commConfig_);
    }

    std::cout << "[DeviceManager] Loaded " << devices_.size() << " devices" << std::endl;
    return !devices_.empty();
}

bool DeviceManager::ParseCommIni(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return false;

    std::string line;
    while (std::getline(file, line)) {
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
        if (line.empty() || line[0] == '[' || line[0] == '#') continue;

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);

        if (key == "connection_retry_delay_ms") commConfig_.connection_retry_delay_ms = std::stoi(val);
        else if (key == "ok_response_timeout_ms") commConfig_.ok_response_timeout_ms = std::stoi(val);
        else if (key == "completion_timeout_ms") commConfig_.completion_timeout_ms = std::stoi(val);
    }
    return true;
}

bool DeviceManager::ConnectAll() {
    bool allOk = true;
    for (auto& [name, conn] : devices_) {
        if (!conn->Connect()) {
            std::cerr << "[DeviceManager] Failed to connect: " << name << std::endl;
            allOk = false;
        }
    }
    return allOk;
}

void DeviceManager::DisconnectAll() {
    for (auto& [name, conn] : devices_) {
        conn->Disconnect();
    }
}

bool DeviceManager::ConnectDevice(const std::string& name) {
    auto dev = FindDevice(name);
    if (!dev) return false;
    return dev->Connect();
}

bool DeviceManager::IsConnected(const std::string& name) const {
    auto dev = FindDevice(name);
    return dev && dev->IsConnected();
}

CommandResponse DeviceManager::SendCommand(const std::string& deviceName, const std::string& command) {
    auto dev = FindDevice(deviceName);
    if (!dev) return {CommandResult::FAILED, "Device not found: " + deviceName};
    return dev->SendCommand(command);
}

std::vector<std::string> DeviceManager::GetDeviceNames() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : devices_) {
        names.push_back(name);
    }
    return names;
}

std::shared_ptr<SocketConnection> DeviceManager::FindDevice(const std::string& name) const {
    auto it = devices_.find(name);
    if (it == devices_.end()) return nullptr;
    return it->second;
}

} // namespace Simtos
