#include "../include/StateMachine.h"
#include <chrono>

namespace Simtos {

static const char* StateToString(SystemState s) {
    switch (s) {
        case SystemState::IDLE:                  return "IDLE";
        case SystemState::ROBOT1_PICK_AND_PLACE: return "ROBOT1_PICK_AND_PLACE";
        case SystemState::SERVO_ROTATE:          return "SERVO_ROTATE";
        case SystemState::ROBOT2_INSERT:         return "ROBOT2_INSERT";
        case SystemState::ROBOT1_RETURN:         return "ROBOT1_RETURN";
        case SystemState::CYCLE_COMPLETE:        return "CYCLE_COMPLETE";
        case SystemState::ERROR:                 return "ERROR";
    }
    return "UNKNOWN";
}

// ========================================
// ConditionController
// ========================================
ConditionController::ConditionController(DeviceManager* devices)
    : devices_(devices) {}

bool ConditionController::Robot1_Ready() {
    return devices_->IsConnected("Pick_Place_Robot");
}

bool ConditionController::ErrorDetected() {
    return !devices_->IsConnected("Pick_Place_Robot") ||
           !devices_->IsConnected("Endoscope_Robot");
}

// ========================================
// ActionController
// ========================================
ActionController::ActionController(DeviceManager* devices, PlcConnection* plc)
    : devices_(devices), plc_(plc) {}

bool ActionController::Robot1_PickToServo(int posIndex) {
    std::string cmdKey = "PP_Pos" + std::to_string(posIndex) + "_Pick_To_Servo";
    auto res = devices_->SendCommand("Pick_Place_Robot", PP_COMMANDS.at(cmdKey));
    if (res.result != CommandResult::SUCCESS) return false;

    std::cout << "[Action] Robot1: Pos" << posIndex << " -> Servo complete" << std::endl;
    return true;
}

bool ActionController::Robot1_ServoToPos(int posIndex) {
    std::string cmdKey = "PP_Servo_To_Pos" + std::to_string(posIndex);
    auto res = devices_->SendCommand("Pick_Place_Robot", PP_COMMANDS.at(cmdKey));
    if (res.result != CommandResult::SUCCESS) return false;

    std::cout << "[Action] Robot1: Servo -> Pos" << posIndex << " complete" << std::endl;
    return true;
}

bool ActionController::Servo_Rotate_Socket(int posIndex) {
    std::string cmdKey = "SERVO_Rotate_" + std::to_string(posIndex);
    auto res = devices_->SendCommand("Servo_Motor", SERVO_COMMANDS.at(cmdKey));
    if (res.result != CommandResult::SUCCESS) return false;

    std::cout << "[Action] Servo(Socket) rotated to position " << posIndex << std::endl;
    return true;
}

bool ActionController::Servo_Rotate_PLC(int posIndex) {
    if (!plc_) return false;

    std::string cmdKey = "SERVO_Rotate_" + std::to_string(posIndex);
    std::string ackKey = cmdKey + "_Ack";

    if (!plc_->SendAndWaitAck(cmdKey, ackKey)) {
        std::cerr << "[Action] Servo(PLC) rotation failed at position " << posIndex << std::endl;
        return false;
    }

    std::cout << "[Action] Servo(PLC) rotated to position " << posIndex << std::endl;
    return true;
}

bool ActionController::Robot2_Insert(int posIndex) {
    std::string cmdKey = "ER_Insert_" + std::to_string(posIndex);
    auto res = devices_->SendCommand("Endoscope_Robot", ER_COMMANDS.at(cmdKey));
    if (res.result != CommandResult::SUCCESS) return false;

    std::cout << "[Action] Robot2 insert at position " << posIndex << " complete" << std::endl;
    return true;
}

// ========================================
// StateMachine
// ========================================
StateMachine::StateMachine(DeviceManager& devices, PlcConnection* plc)
    : devices_(devices)
    , plc_(plc)
    , conditions_(&devices)
    , actions_(&devices, plc) {}

StateMachine::~StateMachine() {
    Stop();
}

void StateMachine::Start() {
    if (running_) return;
    running_ = true;
    workerThread_ = std::thread(&StateMachine::Run, this);
    std::cout << "[StateMachine] Started (servo via "
              << (usePlcForServo_ ? "PLC" : "Socket") << ")" << std::endl;
}

void StateMachine::Stop() {
    running_ = false;
    if (workerThread_.joinable()) {
        workerThread_.join();
    }
    std::cout << "[StateMachine] Stopped" << std::endl;
}

void StateMachine::SetState(SystemState state) {
    currentState_ = state;
    std::cout << "[StateMachine] State -> " << StateToString(state) << std::endl;
}

void StateMachine::Run() {
    while (running_) {
        Update();
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
}

void StateMachine::Update() {
    switch (currentState_.load()) {

        case SystemState::IDLE:
            if (conditions_.Robot1_Ready()) {
                currentServoPos_ = 0;
                currentItemPos_ = 1;
                SetState(SystemState::ROBOT1_PICK_AND_PLACE);
            }
            break;

        case SystemState::ROBOT1_PICK_AND_PLACE:
            if (actions_.Robot1_PickToServo(currentItemPos_)) {
                SetState(SystemState::SERVO_ROTATE);
            } else {
                SetState(SystemState::ERROR);
            }
            break;

        case SystemState::SERVO_ROTATE:
            currentServoPos_++;
            {
                bool ok = usePlcForServo_
                    ? actions_.Servo_Rotate_PLC(currentServoPos_)
                    : actions_.Servo_Rotate_Socket(currentServoPos_);
                if (ok) {
                    SetState(SystemState::ROBOT2_INSERT);
                } else {
                    SetState(SystemState::ERROR);
                }
            }
            break;

        case SystemState::ROBOT2_INSERT:
            if (actions_.Robot2_Insert(currentServoPos_)) {
                if (currentServoPos_ < servoRotationCount_) {
                    SetState(SystemState::SERVO_ROTATE);
                } else {
                    SetState(SystemState::ROBOT1_RETURN);
                }
            } else {
                SetState(SystemState::ERROR);
            }
            break;

        case SystemState::ROBOT1_RETURN:
            if (actions_.Robot1_ServoToPos(currentItemPos_)) {
                SetState(SystemState::CYCLE_COMPLETE);
            } else {
                SetState(SystemState::ERROR);
            }
            break;

        case SystemState::CYCLE_COMPLETE:
            std::cout << "[StateMachine] === Cycle complete (Pos"
                      << currentItemPos_ << ") ===" << std::endl;
            currentItemPos_ = (currentItemPos_ == 1) ? 2 : 1;
            SetState(SystemState::IDLE);
            break;

        case SystemState::ERROR:
            std::cerr << "[StateMachine] ERROR - waiting for recovery" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(3));
            if (!conditions_.ErrorDetected()) {
                SetState(SystemState::IDLE);
            }
            break;
    }
}

} // namespace Simtos
