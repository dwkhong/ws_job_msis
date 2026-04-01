#ifndef SIMTOS_STATE_MACHINE_H
#define SIMTOS_STATE_MACHINE_H

#include "DeviceManager.h"
#include "Commands.h"
#include <atomic>
#include <thread>
#include <iostream>

namespace Simtos {

enum class SystemState {
    IDLE,
    ROBOT1_PICK_AND_PLACE,   // 1번 로봇: 물건 집어서 서보에 올림
    SERVO_ROTATE,            // 서보: 삽입 위치로 회전
    ROBOT2_INSERT,           // 2번 로봇: 내시경 삽입
    ROBOT1_RETURN,           // 1번 로봇: 서보에서 물건 회수해서 원래 위치에 놓음
    CYCLE_COMPLETE,
    ERROR
};

class ConditionController {
public:
    ConditionController(DeviceManager* devices);
    bool Robot1_Ready();
    bool ErrorDetected();
private:
    DeviceManager* devices_;
};

class ActionController {
public:
    ActionController(DeviceManager* devices);

    bool Robot1_PickToServo(int posIndex);  // posIndex 위치에서 집어서 서보에 올림
    bool Robot1_ServoToPos(int posIndex);   // 서보에서 집어서 posIndex 위치에 놓음
    bool Servo_Rotate(int posIndex);        // 서보 posIndex 위치로 회전
    bool Robot2_Insert(int posIndex);       // 내시경 삽입

private:
    DeviceManager* devices_;
};

class StateMachine {
public:
    StateMachine(DeviceManager& devices);
    ~StateMachine();

    void Start();
    void Stop();
    bool IsRunning() const { return running_.load(); }
    SystemState GetState() const { return currentState_.load(); }

    void SetServoRotationCount(int count) { servoRotationCount_ = count; }

private:
    void Run();
    void Update();
    void SetState(SystemState state);

    DeviceManager& devices_;
    ConditionController conditions_;
    ActionController actions_;

    std::atomic<SystemState> currentState_{SystemState::IDLE};
    std::atomic<bool> running_{false};
    std::thread workerThread_;

    int servoRotationCount_ = 5;   // 서보 회전 횟수
    int currentServoPos_ = 0;      // 현재 서보 위치
    int currentItemPos_ = 1;       // 현재 아이템 위치 (1 또는 2)
};

} // namespace Simtos

#endif // SIMTOS_STATE_MACHINE_H
