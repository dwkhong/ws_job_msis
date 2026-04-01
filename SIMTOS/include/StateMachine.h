#ifndef SIMTOS_STATE_MACHINE_H
#define SIMTOS_STATE_MACHINE_H

#include "DeviceManager.h"
#include "PlcConnection.h"
#include "Commands.h"
#include <atomic>
#include <thread>
#include <iostream>

namespace Simtos {

enum class SystemState {
    IDLE,
    ROBOT1_PICK_AND_PLACE,
    SERVO_ROTATE,
    ROBOT2_INSERT,
    ROBOT1_RETURN,
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
    ActionController(DeviceManager* devices, PlcConnection* plc);

    bool Robot1_PickToServo(int posIndex);
    bool Robot1_ServoToPos(int posIndex);
    bool Robot2_Insert(int posIndex);

    // 서보: 소켓 방식
    bool Servo_Rotate_Socket(int posIndex);
    // 서보: PLC 방식 (WOOSU_FRAMEWORK 패턴)
    bool Servo_Rotate_PLC(int posIndex);

private:
    DeviceManager* devices_;
    PlcConnection* plc_;
};

class StateMachine {
public:
    StateMachine(DeviceManager& devices, PlcConnection* plc = nullptr);
    ~StateMachine();

    void Start();
    void Stop();
    bool IsRunning() const { return running_.load(); }
    SystemState GetState() const { return currentState_.load(); }

    void SetServoRotationCount(int count) { servoRotationCount_ = count; }
    void SetUsePlcForServo(bool use) { usePlcForServo_ = use; }

private:
    void Run();
    void Update();
    void SetState(SystemState state);

    DeviceManager& devices_;
    PlcConnection* plc_;
    ConditionController conditions_;
    ActionController actions_;

    std::atomic<SystemState> currentState_{SystemState::IDLE};
    std::atomic<bool> running_{false};
    std::thread workerThread_;

    int servoRotationCount_ = 5;
    int currentServoPos_ = 0;
    int currentItemPos_ = 1;
    bool usePlcForServo_ = false;  // false=소켓, true=PLC
};

} // namespace Simtos

#endif // SIMTOS_STATE_MACHINE_H
