#include "include/DeviceManager.h"
#include "include/StateMachine.h"

#include <iostream>
#include <csignal>
#include <atomic>

static std::atomic<bool> g_running{true};

void signalHandler(int) {
    std::cout << "\n[SIMTOS] Shutdown requested..." << std::endl;
    g_running = false;
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signalHandler);

    std::string configDir = "config";
    if (argc >= 2) configDir = argv[1];

    std::cout << "========================================" << std::endl;
    std::cout << "  SIMTOS - Robot Control Framework" << std::endl;
    std::cout << "========================================" << std::endl;

    // 1) 디바이스 매니저 초기화
    Simtos::DeviceManager deviceManager;
    if (!deviceManager.LoadConfig(configDir + "/robots.ini", configDir + "/comm_config.ini")) {
        std::cerr << "[SIMTOS] Config load failed" << std::endl;
        return 1;
    }

    std::cout << "[SIMTOS] Devices:" << std::endl;
    for (const auto& name : deviceManager.GetDeviceNames()) {
        std::cout << "  - " << name << std::endl;
    }

    // 2) 모든 디바이스 연결
    if (!deviceManager.ConnectAll()) {
        std::cerr << "[SIMTOS] Not all devices connected." << std::endl;
        return 1;
    }

    // 3) 스테이트 머신 시작
    Simtos::StateMachine stateMachine(deviceManager);
    stateMachine.SetServoRotationCount(5);  // 서보 5번 회전
    stateMachine.Start();

    // Ctrl+C 대기
    while (g_running && stateMachine.IsRunning()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // 4) 종료
    stateMachine.Stop();
    deviceManager.DisconnectAll();

    std::cout << "[SIMTOS] Shutdown complete" << std::endl;
    return 0;
}
