#include "../include/StateMachine.h"
#include <chrono>

namespace Simtos {

static const char* StateToString(SystemState s) {
    switch (s) {
        case SystemState::IDLE:                return "IDLE";
        case SystemState::ROBOT1_PICK_AND_PLACE: return "ROBOT1_PICK_AND_PLACE";
        case SystemState::SERVO_ROTATE:        return "SERVO_ROTATE";
        case SystemState::ROBOT2_INSERT:       return "ROBOT2_INSERT";
        case SystemState::ROBOT1_RETURN:       return "ROBOT1_RETURN";
        case SystemState::CYCLE_COMPLETE:      return "CYCLE_COMPLETE";
        case SystemState::ERROR:               return "ERROR";
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
           !devices_->IsConnected("Endoscope_Robot") ||
           !devices_->IsConnected("Servo_Motor");
}

// ========================================
// ActionController
// ========================================
ActionController::ActionController(DeviceManager* devices)
    : devices_(devices) {}

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

bool ActionController::Servo_Rotate(int posIndex) {
    std::string cmdKey = "SERVO_Rotate_" + std::to_string(posIndex);
    auto res = devices_->SendCommand("Servo_Motor", SERVO_COMMANDS.at(cmdKey));
    if (res.result != CommandResult::SUCCESS) return false;

    std::cout << "[Action] Servo rotated to position " << posIndex << std::endl;
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
StateMachine::StateMachine(DeviceManager& devices)
    : devices_(devices)
    , conditions_(&devices)
    , actions_(&devices) {}

StateMachine::~StateMachine() {
    Stop();
}

void StateMachine::Start() {
    if (running_) return;
    running_ = true;
    workerThread_ = std::thread(&StateMachine::Run, this);
    std::cout << "[StateMachine] Started" << std::endl;
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
                currentItemPos_ = 1;  // 1번위치부터 시작
                SetState(SystemState::ROBOT1_PICK_AND_PLACE);
            }
            break;

        case SystemState::ROBOT1_PICK_AND_PLACE:
            // 1번 로봇: currentItemPos_ 위치에서 집어서 서보에 올림
            if (actions_.Robot1_PickToServo(currentItemPos_)) {
                SetState(SystemState::SERVO_ROTATE);
            } else {
                SetState(SystemState::ERROR);
            }
            break;

        case SystemState::SERVO_ROTATE:
            // 서보: 다음 위치로 회전
            currentServoPos_++;
            if (actions_.Servo_Rotate(currentServoPos_)) {
                SetState(SystemState::ROBOT2_INSERT);
            } else {
                SetState(SystemState::ERROR);
            }
            break;

        case SystemState::ROBOT2_INSERT:
            // 2번 로봇: 현재 서보 위치에서 내시경 삽입
            if (actions_.Robot2_Insert(currentServoPos_)) {
                if (currentServoPos_ < servoRotationCount_) {
                    // 아직 남은 위치 있음 -> 서보 다시 회전
                    SetState(SystemState::SERVO_ROTATE);
                } else {
                    // 모든 위치 완료 -> 물건 회수
                    SetState(SystemState::ROBOT1_RETURN);
                }
            } else {
                SetState(SystemState::ERROR);
            }
            break;

        case SystemState::ROBOT1_RETURN:
            // 1번 로봇: 서보에서 집어서 원래 위치에 놓음
            if (actions_.Robot1_ServoToPos(currentItemPos_)) {
                SetState(SystemState::CYCLE_COMPLETE);
            } else {
                SetState(SystemState::ERROR);
            }
            break;

        case SystemState::CYCLE_COMPLETE:
            std::cout << "[StateMachine] === Cycle complete (item from Pos"
                      << currentItemPos_ << ") ===" << std::endl;
            // 다음 아이템 위치로 전환 (1 <-> 2)
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
