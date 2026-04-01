#include "include/DeviceManager.h"
#include "include/PlcConnection.h"
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

    // 1) 디바이스 매니저 초기화 (로봇 소켓 통신)
    Simtos::DeviceManager deviceManager;
    if (!deviceManager.LoadConfig(configDir + "/robots.ini", configDir + "/comm_config.ini")) {
        std::cerr << "[SIMTOS] Config load failed" << std::endl;
        return 1;
    }

    std::cout << "[SIMTOS] Devices:" << std::endl;
    for (const auto& name : deviceManager.GetDeviceNames()) {
        std::cout << "  - " << name << std::endl;
    }

    // 2) PLC 연결 (서보모터용, 선택)
    Simtos::PlcConnection plc;
    bool usePlc = false;

    // PLC config 파일이 있으면 PLC 모드
    std::ifstream plcFile(configDir + "/plc_config.ini");
    if (plcFile.good()) {
        plcFile.close();
        plc.LoadAddressMap(configDir + "/plc_address_map.ini");

        // TODO: plc_config.ini에서 IP/포트 읽기
        // 지금은 하드코딩
        if (plc.Connect("192.168.0.100", 5000)) {
            usePlc = true;
            std::cout << "[SIMTOS] Servo control via PLC" << std::endl;
        } else {
            std::cout << "[SIMTOS] PLC connection failed, falling back to Socket" << std::endl;
        }
    }

    // 3) 로봇 연결 (연결 가능한 것만 연결)
    deviceManager.ConnectAll();
    std::cout << "[SIMTOS] Connected devices:" << std::endl;
    for (const auto& name : deviceManager.GetDeviceNames()) {
        std::cout << "  - " << name << ": "
                  << (deviceManager.IsConnected(name) ? "OK" : "FAIL") << std::endl;
    }

    // 4) 스테이트 머신
    Simtos::StateMachine stateMachine(deviceManager, usePlc ? &plc : nullptr);
    stateMachine.SetServoRotationCount(5);
    stateMachine.SetUsePlcForServo(usePlc);
    stateMachine.Start();

    // Ctrl+C 대기
    while (g_running && stateMachine.IsRunning()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // 5) 종료
    stateMachine.Stop();
    deviceManager.DisconnectAll();
    plc.Disconnect();

    std::cout << "[SIMTOS] Shutdown complete" << std::endl;
    return 0;
}
