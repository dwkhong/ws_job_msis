import sys
import time

# ✅ Robot.cp39-win_amd64.pyd 폴더를 Python 경로에 추가
sys.path.insert(
    0,
    r"C:\Users\rhdeh\ws_job_msis\amr_project\driver\fairino-python-sdk-main\windows\fairino\build\lib.win-amd64-cpython-39"
)
import Robot

ROBOT_IP = "192.168.58.3"

# -------------------------
# Format / Parse
# -------------------------
def fmt_pose6(p):
    if not isinstance(p, (list, tuple)) or len(p) < 6:
        return str(p)
    x, y, z, rx, ry, rz = p[:6]
    return f"[x,y,z,rx,ry,rz]=[{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}]"

def fmt_list(lst, n=6):
    if not isinstance(lst, (list, tuple)) or len(lst) < n:
        return str(lst)
    return "[" + ", ".join(f"{float(v):.3f}" for v in lst[:n]) + "]"

def parse_n(line: str, n: int):
    s = line.replace(",", " ").strip()
    parts = [p for p in s.split() if p]
    if len(parts) != n:
        raise ValueError(f"{n}개가 아님: {len(parts)}개 입력됨 -> {parts}")
    return [float(p) for p in parts]

def ensure_len7(arr6_or_7):
    """문서에 [0..0] 7개 배열이 등장하는 인자용: 6개면 뒤에 0.0을 붙여 7개로 맞춤"""
    if not isinstance(arr6_or_7, (list, tuple)):
        raise ValueError("not list/tuple")
    a = list(arr6_or_7)
    if len(a) == 6:
        a.append(0.0)
    if len(a) < 7:
        raise ValueError(f"len < 7: {len(a)}")
    return a[:7]

# -------------------------
# Actual read
# -------------------------
def show_actual(robot, flag=1):
    try:
        err, tcp = robot.GetActualTCPPose(flag=flag)
        if err == 0:
            print("[ACTUAL TCP ]", fmt_pose6(tcp))
        else:
            print(f"[ACTUAL TCP ] FAIL err={err}, tcp={tcp}")
    except Exception as e:
        print(f"[ACTUAL TCP ] EXCEPTION: {e}")

    try:
        ret, j = robot.GetActualJointPosDegree(flag)
        if ret == 0:
            print("[ACTUAL JNT ]", fmt_list(j, 6))
        else:
            print(f"[ACTUAL JNT ] FAIL ret={ret}, j={j}")
    except Exception as e:
        print(f"[ACTUAL JNT ] EXCEPTION: {e}")

# -------------------------
# Menu
# -------------------------
def prompt_menu():
    print("\n=======================================")
    print("  1 : 현재 위치 출력 (TCP/Joints)")
    print("  2 : MoveL    (x y z rx ry rz) 6개 입력 -> Linear")
    print("  3 : MoveCart (x y z rx ry rz) 6개 입력 -> Cartesian PTP")
    print("  4 : MoveJ    (j1 j2 j3 j4 j5 j6) 6개 입력 -> Joint PTP")
    print("  5 : MoveC    (path pose 6개 + target pose 6개) -> Circular")
    print("  9 : 파라미터 보기/수정")
    print("  q : 종료")
    print("=======================================")
    return input("입력 > ").strip().lower()

# -------------------------
# Main
# -------------------------
def main():
    print(f"[INFO] Connecting... {ROBOT_IP}")
    robot = Robot.RPC(ROBOT_IP)
    print("[OK] Connected ✅")

    # -------------------------
    # ✅ 기본값: 문서 디폴트 기준
    # -------------------------
    tool = 0
    user = 0

    vel = 20.0
    acc = 0.0          # 문서: not open yet, default 0.0
    ovl = 100.0

    # MoveL
    blendR = -1.0      # [-1 blocking] or [0~1000 radius mm]
    blendMode = 0      # 0 internal cutting, 1 corner
    exaxis_pos = [0.0, 0.0, 0.0, 0.0]
    search = 0
    offset_flag = 0
    offset_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    config = -1
    velAccParamMode = 0
    overSpeedStrategy = 0
    speedPercent = 10

    # MoveCart / MoveJ
    blendT = -1.0      # [-1 blocking] or [0~500 ms]

    try:
        while True:
            cmd = prompt_menu()

            if cmd == "1":
                show_actual(robot, flag=1)

            elif cmd == "2":
                print("\n--- 현재 위치 ---")
                show_actual(robot, flag=1)

                line = input("MoveL target (x y z rx ry rz) > ")
                try:
                    pose6 = parse_n(line, 6)
                except Exception as e:
                    print(f"[WARN] 입력 오류: {e}")
                    continue

                desc_pos = ensure_len7(pose6)               # 문서 프로토타입 배열 길이 맞춤
                joint_pos = [0.0]*7                         # 디폴트: IK로 풀이
                print("[TARGET]", fmt_pose6(desc_pos))

                try:
                    rtn = robot.MoveL(
                        desc_pos=desc_pos,
                        tool=tool,
                        user=user,
                        joint_pos=joint_pos,
                        vel=vel,
                        acc=acc,
                        ovl=ovl,
                        blendR=blendR,
                    )
                    print(f"[RET] MoveL errcode: {rtn}")
                except Exception as e:
                    print(f"[ERROR] MoveL 예외: {e}")

            elif cmd == "3":
                print("\n--- 현재 위치 ---")
                show_actual(robot, flag=1)

                line = input("MoveCart target (x y z rx ry rz) > ")
                try:
                    pose6 = parse_n(line, 6)
                except Exception as e:
                    print(f"[WARN] 입력 오류: {e}")
                    continue

                desc_pos = ensure_len7(pose6)
                print("[TARGET]", fmt_pose6(desc_pos))

                try:
                    rtn = robot.MoveCart(
                        desc_pos=desc_pos,
                        tool=tool,
                        user=user,
                        vel=vel,
                        acc=acc,
                        ovl=ovl,
                        blendT=blendT,
                        config=config
                    )
                    print(f"[RET] MoveCart errcode: {rtn}")
                except Exception as e:
                    print(f"[ERROR] MoveCart 예외: {e}")

            elif cmd == "4":
                print("\n--- 현재 위치 ---")
                show_actual(robot, flag=1)

                line = input("MoveJ target joints (j1 j2 j3 j4 j5 j6) > ")
                try:
                    j6 = parse_n(line, 6)
                except Exception as e:
                    print(f"[WARN] 입력 오류: {e}")
                    continue

                joint_pos = ensure_len7(j6)     # 문서 배열 길이 맞춤
                desc_pos  = [0.0]*7             # 디폴트: FK로 계산(문서 설명)
                print("[JOINT TARGET]", fmt_list(joint_pos, 6))

                try:
                    rtn = robot.MoveJ(
                        joint_pos=joint_pos,
                        tool=tool,
                        user=user,
                        desc_pos=desc_pos,
                        vel=vel,
                        acc=acc,
                        ovl=ovl,
                        exaxis_pos=exaxis_pos,
                        blendT=blendT,
                        offset_flag=offset_flag,
                        offset_pos=offset_pos
                    )
                    print(f"[RET] MoveJ errcode: {rtn}")
                except Exception as e:
                    print(f"[ERROR] MoveJ 예외: {e}")

            elif cmd == "5":
                print("\n--- 현재 위치 ---")
                show_actual(robot, flag=1)

                line_p = input("MoveC PATH  pose (x y z rx ry rz) > ")
                line_t = input("MoveC TARGET pose (x y z rx ry rz) > ")
                try:
                    p6 = parse_n(line_p, 6)
                    t6 = parse_n(line_t, 6)
                except Exception as e:
                    print(f"[WARN] 입력 오류: {e}")
                    continue

                desc_pos_p = ensure_len7(p6)
                desc_pos_t = ensure_len7(t6)

                joint_pos_p = [0.0]*7  # 디폴트: IK
                joint_pos_t = [0.0]*7  # 디폴트: IK

                print("[PATH  ]", fmt_pose6(desc_pos_p))
                print("[TARGET]", fmt_pose6(desc_pos_t))

                try:
                    rtn = robot.MoveC(
                        desc_pos_p=desc_pos_p,
                        tool_p=tool,
                        user_p=user,
                        desc_pos_t=desc_pos_t,
                        tool_t=tool,
                        user_t=user,
                        joint_pos_p=joint_pos_p,
                        joint_pos_t=joint_pos_t,
                        vel_p=vel,
                        acc_p=acc,
                        exaxis_pos_p=exaxis_pos,
                        offset_flag_p=offset_flag,
                        offset_pos_p=offset_pos,
                        vel_t=vel,
                        acc_t=acc,
                        exaxis_pos_t=exaxis_pos,
                        offset_flag_t=offset_flag,
                        offset_pos_t=offset_pos,
                        ovl=ovl,
                        blendR=blendR,
                        config=config,
                        velAccParamMode=velAccParamMode
                    )
                    print(f"[RET] MoveC errcode: {rtn}")
                except Exception as e:
                    print(f"[ERROR] MoveC 예외: {e}")

            elif cmd == "9":
                print("\n--- 현재 파라미터 ---")
                print(f"tool={tool}, user={user}")
                print(f"vel={vel}, acc={acc}, ovl={ovl}")
                print(f"blendR={blendR}, blendT={blendT}, blendMode={blendMode}")
                print(f"config={config}, velAccParamMode={velAccParamMode}")
                print(f"search={search}, offset_flag={offset_flag}, speedPercent={speedPercent}, overSpeedStrategy={overSpeedStrategy}")
                print(f"exaxis_pos={exaxis_pos}")
                print(f"offset_pos={offset_pos}")

                print("\n수정 예시: vel=10  또는  tool=1  또는  blendR=-1  (엔터=취소)")
                line = input("수정 입력 (key=value) > ").strip()
                if not line:
                    continue
                if "=" not in line:
                    print("[WARN] 형식은 key=value")
                    continue
                k, v = [x.strip() for x in line.split("=", 1)]

                try:
                    if k in ("tool", "user", "blendMode", "search", "offset_flag", "config", "velAccParamMode", "overSpeedStrategy", "speedPercent"):
                        val = int(float(v))
                    else:
                        val = float(v)

                    if k == "tool": tool = val
                    elif k == "user": user = val
                    elif k == "vel": vel = float(val)
                    elif k == "acc": acc = float(val)
                    elif k == "ovl": ovl = float(val)
                    elif k == "blendR": blendR = float(val)
                    elif k == "blendT": blendT = float(val)
                    elif k == "blendMode": blendMode = int(val)
                    elif k == "search": search = int(val)
                    elif k == "offset_flag": offset_flag = int(val)
                    elif k == "config": config = int(val)
                    elif k == "velAccParamMode": velAccParamMode = int(val)
                    elif k == "overSpeedStrategy": overSpeedStrategy = int(val)
                    elif k == "speedPercent": speedPercent = int(val)
                    else:
                        print("[WARN] 지원 안 하는 key")
                except Exception as e:
                    print(f"[WARN] 수정 실패: {e}")

            elif cmd == "q":
                print("[EXIT]")
                break

            else:
                print("[WARN] 잘못된 입력")

            time.sleep(0.05)

    finally:
        try:
            robot.CloseRPC()
        except Exception:
            pass

if __name__ == "__main__":
    main()


