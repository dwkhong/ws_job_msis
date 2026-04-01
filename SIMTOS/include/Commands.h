#pragma once
#include <string>
#include <map>

// ========================================
// SIMTOS Robot Command Definitions
// ========================================
// 각 명령 코드는 로봇 컨트롤러 안에 이미 티칭된 동작에 대응
// PC에서 소켓으로 코드를 보내면 로봇이 해당 동작 실행 후 "done" 반환

// Pick & Place Robot (1번 로봇)
static const std::map<std::string, std::string> PP_COMMANDS = {
    {"PP_Pos1_Pick_To_Servo",     "1001"},  // 1번위치에서 집어서 서보 위에 올림
    {"PP_Servo_To_Pos1",          "1002"},  // 서보에서 집어서 1번위치에 놓음
    {"PP_Pos2_Pick_To_Servo",     "1003"},  // 2번위치에서 집어서 서보 위에 올림
    {"PP_Servo_To_Pos2",          "1004"},  // 서보에서 집어서 2번위치에 놓음
};

// Endoscope Robot (2번 로봇) - 내시경 삽입
static const std::map<std::string, std::string> ER_COMMANDS = {
    {"ER_Insert_1",               "2001"},  // 1번 위치 삽입 (티칭파일에 찌르기 동작 포함)
    {"ER_Insert_2",               "2002"},  // 2번 위치 삽입
    {"ER_Insert_3",               "2003"},  // 3번 위치 삽입
    {"ER_Insert_4",               "2004"},  // 4번 위치 삽입
    {"ER_Insert_5",               "2005"},  // 5번 위치 삽입
};

// Servo Motor - 물건 회전
static const std::map<std::string, std::string> SERVO_COMMANDS = {
    {"SERVO_Rotate_1",            "3001"},  // 1번 위치로 회전
    {"SERVO_Rotate_2",            "3002"},  // 2번 위치로 회전
    {"SERVO_Rotate_3",            "3003"},  // 3번 위치로 회전
    {"SERVO_Rotate_4",            "3004"},  // 4번 위치로 회전
    {"SERVO_Rotate_5",            "3005"},  // 5번 위치로 회전
};
