import os
import sys
import time
import math
import threading
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

from box_test_6 import main as measure_main

# ✅ Robot.cp39-win_amd64.pyd 폴더를 Python 경로에 추가
sys.path.insert(
    0,
    r"C:\Users\rhdeh\ws_job_msis\amr_project\driver\fairino-python-sdk-main\windows\fairino\build\lib.win-amd64-cpython-39"
)
import Robot

ROBOT_IP = "192.168.0.15"

# ============================================================
# ✅ SPEED CONFIG (여기서 %로 조절)
# ============================================================
MOVE_CART_VEL_DEFAULT = 70.0
MOVE_CART_VEL_FALLBACKS = [10.0, 5.0, 3.0]     # 실패/112 나오면 순차 적용
MOVE_CART_VEL_LIST = [MOVE_CART_VEL_DEFAULT] + MOVE_CART_VEL_FALLBACKS

MOVEJ_VEL_J6 = 70.0
MOVEJ_BLENDT_J6 = -1.0

MOVEJ_VEL_RETURN = 70.0
MOVEJ_BLENDT_RETURN = -1.0

# ✅ 11번용 MoveJ 속도
MOVEJ_VEL_WP11 = 70.0
MOVEJ_BLENDT_WP11 = -1.0

# -------------------------
# Step config (7번용 유지)
# -------------------------
STEP_SCALE_DEFAULT = 0.3
X_SCALE_MULT = 2.0

# -------------------------
# ✅ 2-Phase approach
# -------------------------
Z_HOLD_OFFSET_MM = 70.0
XY_TOL_MM = 1
Z_TOL_MM  = 2

# -------------------------
# ✅ 6번(J6 회전)
# -------------------------
ANGLE_TO_J6_SIGN = +1.0
J6_MAX_STEP_DEG  = 45.0

# -------------------------
# ✅ 7/9 step try config (7번용 유지)
# -------------------------
STEP_TRY_LIST_DEFAULT = [STEP_SCALE_DEFAULT, 0.05, 0.02, 0.01]

# -------------------------
# ✅ 9번(자동) safety
# -------------------------
AUTO_MAX_SECONDS = 60.0

# ============================================================
# ✅ GRIPPER CONFIG (10번/9번 후 자동닫기)
# ============================================================
GRIPPER_INDEX = 1
GRIPPER_MAX_TIME = 30000
GRIPPER_SPEED = 90
GRIPPER_FORCE = 50
GRIPPER_BLOCK = 1

GRIP_OPEN_POS = 100
GRIP_CLOSE_POS = 60

# ============================================================
# ✅ 11번 스택 사이클 설정
# ============================================================
WP11_A_POSE = [76.752, 218.531, 466.100, -176.747, 11.609, -128.123]
WP11_A_JOINT = [-134.487, -111.775, 75.336, -65.453, -91.956, 83.763]

WP11_DROP_BASE_POSE = [185.145, 312.801, 228.808, -177.820, 1.463, -127.365]

STACK_Z_STEP_MM = 48.0
STACK_Z_MAX_MM = 600.0  # None이면 제한 없음

# ============================================================
# ✅ ALWAYS RECORD CONFIG (프로그램 실행 중 상시 녹화)
# ============================================================
ALWAYS_RECORD = True
REC_DIR = "recordings_live"
REC_WIDTH = 640
REC_HEIGHT = 480
REC_FPS = 30
REC_FOURCC = "mp4v"
REC_SEGMENT_SEC = 300  # 5분마다 파일 분할(너무 커지는 거 방지)
REC_SHOW_PREVIEW = True
REC_PREVIEW_WIN = "LIVE REC (overlay XYZ/angle)"

# 오버레이 폰트
OVL_FONT_SCALE = 0.55
OVL_THICKNESS = 2


# -------------------------
# Utils
# -------------------------
def fmt_pose6(pose):
    if not isinstance(pose, (list, tuple)) or len(pose) < 6:
        return str(pose)
    x, y, z, rx, ry, rz = pose[:6]
    return f"[x,y,z,rx,ry,rz]=[{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}]"


def fmt_joint(j):
    if not isinstance(j, (list, tuple)):
        return str(j)
    return "[" + ", ".join(f"{float(v):.3f}" for v in j[:6]) + "]"


def ensure_pose6(p):
    if not isinstance(p, (list, tuple)):
        raise ValueError(f"pose is not list/tuple: {type(p)}")
    if len(p) < 6:
        raise ValueError(f"pose length < 6: {len(p)}")
    return [float(x) for x in p[:6]]


def ensure_joint6(j):
    if not isinstance(j, (list, tuple)):
        raise ValueError(f"joint is not list/tuple: {type(j)}")
    if len(j) < 6:
        raise ValueError(f"joint length < 6: {len(j)}")
    return [float(x) for x in j[:6]]


def joint_delta_str(j_new, j_old):
    if j_new is None or j_old is None:
        return "(delta N/A)"
    d = [float(a) - float(b) for a, b in zip(j_new[:6], j_old[:6])]
    return "[" + ", ".join(f"{v:+.3f}" for v in d) + "]"


def build_target_pose(cur_pose, move_res):
    x, y, z, rx, ry, rz = ensure_pose6(cur_pose)
    dx = float(move_res["move_x_mm"])
    dy = float(move_res["move_y_mm"])
    dz = float(move_res["move_z_mm"])
    return [x + dx, y + dy, z + dz, rx, ry, rz]


def blend_pose(cur_pose, target_pose, scale):
    cur = ensure_pose6(cur_pose)
    tgt = ensure_pose6(target_pose)
    s = float(scale)
    return [cur[i] + (tgt[i] - cur[i]) * s for i in range(6)]


def blend_pose_axis(cur_pose, target_pose, scale_xyz, ori_scale=None):
    cur = ensure_pose6(cur_pose)
    tgt = ensure_pose6(target_pose)

    sx, sy, sz = [float(v) for v in scale_xyz]
    if ori_scale is None:
        ori_scale = sy
    else:
        ori_scale = float(ori_scale)

    out = cur[:]

    for i, s in zip([0, 1, 2], [sx, sy, sz]):
        delta = tgt[i] - cur[i]
        step = delta * s
        if delta >= 0:
            out[i] = min(cur[i] + step, tgt[i])
        else:
            out[i] = max(cur[i] + step, tgt[i])

    for i in [3, 4, 5]:
        out[i] = cur[i] + (tgt[i] - cur[i]) * ori_scale

    return out


def xy_reached(cur_pose6, target_pose6, tol_mm=2.0):
    cur = ensure_pose6(cur_pose6)
    tgt = ensure_pose6(target_pose6)
    return (abs(cur[0] - tgt[0]) <= tol_mm) and (abs(cur[1] - tgt[1]) <= tol_mm)


def z_reached(cur_pose6, z_target, tol_mm=2.0):
    cur = ensure_pose6(cur_pose6)
    return abs(cur[2] - float(z_target)) <= tol_mm


def make_phase0_pose_from_target(target_pose6):
    """target_pose(z=목표) -> phase0_pose(z=목표+Z_HOLD_OFFSET)"""
    t = ensure_pose6(target_pose6)
    p0 = t[:]
    p0[2] = float(t[2]) + float(Z_HOLD_OFFSET_MM)
    return p0


def connect_robot(ip):
    print("[INFO] Connecting to Fairino...")
    rb = Robot.RPC(ip)
    print(rb)
    print("[INFO] Robot connected ✅\n")
    return rb


def safe_call(fn, *args, retry=1, sleep_sec=0.25, reconnect_cb=None, **kwargs):
    last_e = None
    for k in range(retry + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_e = e
            msg = str(e)
            if ("timed out" in msg.lower()) or ("实时数据失败" in msg) or ("timeout" in msg.lower()):
                print(f"[WARN] RPC timeout-like error: {e}")
                if reconnect_cb is not None:
                    try:
                        reconnect_cb()
                    except Exception as e2:
                        print(f"[WARN] reconnect failed: {e2}")
            if k < retry:
                time.sleep(sleep_sec)
                continue
            raise last_e


def read_pose_joint(robot, reconnect=None):
    err_p, pose = safe_call(robot.GetActualTCPPose, flag=1, retry=1, reconnect_cb=reconnect)
    err_j, joint = safe_call(robot.GetActualJointPosDegree, flag=1, retry=1, reconnect_cb=reconnect)
    if err_p != 0 or err_j != 0:
        return (err_p, None), (err_j, None)
    return (0, ensure_pose6(pose)), (0, ensure_joint6(joint))


def has_solution(robot, pose6, cur_joint6, reconnect=None):
    err, ok = safe_call(robot.GetInverseKinHasSolution, 0, pose6, cur_joint6, retry=1, reconnect_cb=reconnect)
    if err != 0:
        return False
    return bool(ok)


def get_ik(robot, pose6, cur_joint6, reconnect=None):
    err, j = safe_call(robot.GetInverseKinRef, 0, pose6, cur_joint6, retry=1, reconnect_cb=reconnect)
    if err != 0:
        return None
    return ensure_joint6(j)


# ============================================================
# ✅ Live Recorder (RealSense) - overlay: moveXYZ/angle만
# ============================================================
def _draw_overlay_move_xyz_angle(img, last_measure):
    """
    last_measure = {"move_x_mm","move_y_mm","move_z_mm","angle_deg"} 형태를 기대.
    없으면 N/A 표시.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = OVL_FONT_SCALE
    th = OVL_THICKNESS

    if not last_measure:
        line1 = "X N/A  Y N/A  Z N/A (mm)"
        line2 = "angle N/A (deg)"
    else:
        try:
            mx = float(last_measure.get("move_x_mm"))
            my = float(last_measure.get("move_y_mm"))
            mz = float(last_measure.get("move_z_mm"))
            ang = float(last_measure.get("angle_deg", 0.0))
            line1 = f"X {mx:+.1f}  Y {my:+.1f}  Z {mz:+.1f} (mm)"
            line2 = f"angle {ang:+.2f} (deg)"
        except Exception:
            line1 = "X ?  Y ?  Z ? (mm)"
            line2 = "angle ? (deg)"

    (w1, h1), _ = cv2.getTextSize(line1, font, fs, th)
    (w2, h2), _ = cv2.getTextSize(line2, font, fs, th)
    w = max(w1, w2)
    h = h1 + h2 + 18

    overlay = img.copy()
    cv2.rectangle(overlay, (6, 6), (6 + w + 12, 6 + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    x, y = 10, 14
    cv2.putText(img, line1, (x, y + 18), font, fs, (255, 255, 255), th, cv2.LINE_AA)
    cv2.putText(img, line2, (x, y + 18 + h1 + 6), font, fs, (255, 255, 255), th, cv2.LINE_AA)


class LiveRecorder:
    """
    - 프로그램 실행 중 계속 녹화
    - cmd=2(측정) 실행 시 카메라 충돌 방지 위해 pause()로 카메라 release
    - segment 시간마다 mp4 파일 분할 저장
    """
    def __init__(self, get_last_measure_callable):
        self.get_last_measure = get_last_measure_callable
        self.stop_evt = threading.Event()
        self.pause_evt = threading.Event()
        self.thread = None

        self.pipeline = None
        self.writer = None
        self.cur_path = None
        self.seg_t0 = None

    def start(self):
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_evt.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self._release_all()

    def pause(self):
        self.pause_evt.set()
        # pause 즉시 카메라/파일 닫아서 measure_main이 잡을 수 있게
        self._release_all()

    def resume(self):
        self.pause_evt.clear()

    def _release_all(self):
        try:
            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
        finally:
            self.pipeline = None

        try:
            if self.writer is not None:
                try:
                    self.writer.release()
                except Exception:
                    pass
        finally:
            self.writer = None
            self.cur_path = None
            self.seg_t0 = None

    def _open_new_segment(self):
        os.makedirs(REC_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cur_path = os.path.abspath(os.path.join(REC_DIR, f"live_{ts}.mp4"))

        fourcc = cv2.VideoWriter_fourcc(*REC_FOURCC)
        self.writer = cv2.VideoWriter(self.cur_path, fourcc, float(REC_FPS), (REC_WIDTH, REC_HEIGHT))
        self.seg_t0 = time.time()

        if not self.writer.isOpened():
            print(f"[REC-WARN] VideoWriter open 실패: {self.cur_path}")
            self.writer = None
        else:
            print(f"[REC] recording -> {self.cur_path}")

    def _ensure_pipeline(self):
        if self.pipeline is not None:
            return True
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, REC_WIDTH, REC_HEIGHT, rs.format.bgr8, REC_FPS)
            self.pipeline.start(config)
            return True
        except Exception as e:
            self.pipeline = None
            print(f"[REC-WARN] RealSense start 실패(장치 busy/미연결 가능): {e}")
            return False

    def _run(self):
        if REC_SHOW_PREVIEW:
            cv2.namedWindow(REC_PREVIEW_WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(REC_PREVIEW_WIN, REC_WIDTH, REC_HEIGHT)

        while not self.stop_evt.is_set():
            if self.pause_evt.is_set():
                time.sleep(0.1)
                continue

            if not self._ensure_pipeline():
                time.sleep(0.3)
                continue

            if self.writer is None:
                self._open_new_segment()

            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                img = np.asanyarray(color_frame.get_data())
                # overlay: moveXYZ/angle
                _draw_overlay_move_xyz_angle(img, self.get_last_measure())

                if REC_SHOW_PREVIEW:
                    cv2.imshow(REC_PREVIEW_WIN, img)
                    # (ESC로 미리보기만 닫고 녹화는 계속)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        cv2.destroyWindow(REC_PREVIEW_WIN)

                if self.writer is not None:
                    self.writer.write(img)

                # segment rotate
                if self.seg_t0 is not None and (time.time() - self.seg_t0) >= float(REC_SEGMENT_SEC):
                    # 다음 파일로 넘어가기
                    if self.writer is not None:
                        self.writer.release()
                    self.writer = None
                    self.cur_path = None
                    self.seg_t0 = None

            except Exception as e:
                # 뻗어도 다시 살아나게
                print(f"[REC-WARN] record loop exception: {e}")
                self._release_all()
                time.sleep(0.3)

        # 종료
        self._release_all()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# ------------------------------------------------------------
# ✅ Gripper Helpers (원본 유지)
# ------------------------------------------------------------
def ensure_gripper_activated(robot, reconnect=None, state=None):
    if state is not None and state.get("gripper_activated", False):
        return True

    print("[GRIP] Activating gripper...")
    try:
        err = safe_call(robot.ActGripper, GRIPPER_INDEX, 1, retry=1, reconnect_cb=reconnect)
    except Exception as e:
        print(f"[GRIP-FAIL] ActGripper exception: {e}")
        return False

    print("[GRIP] ActGripper:", err)
    if err == 0 and state is not None:
        state["gripper_activated"] = True
    time.sleep(0.3)
    return (err == 0)


def gripper_move(robot, pos, reconnect=None):
    try:
        err = safe_call(
            robot.MoveGripper,
            GRIPPER_INDEX,
            int(pos),
            int(GRIPPER_SPEED),
            int(GRIPPER_FORCE),
            int(GRIPPER_MAX_TIME),
            int(GRIPPER_BLOCK),
            0, 0, 0, 0,
            retry=1,
            reconnect_cb=reconnect
        )
        return err
    except Exception as e:
        print(f"[GRIP-FAIL] MoveGripper exception: {e}")
        return -999


def gripper_open(robot, reconnect=None, state=None):
    if not ensure_gripper_activated(robot, reconnect=reconnect, state=state):
        return False
    print("[GRIP] Opening gripper...")
    err = gripper_move(robot, GRIP_OPEN_POS, reconnect=reconnect)
    print("[GRIP] Open retval:", err)
    if err == 0 and state is not None:
        state["gripper_closed"] = False
    time.sleep(0.3)
    return (err == 0)


def gripper_close(robot, reconnect=None, state=None):
    if not ensure_gripper_activated(robot, reconnect=reconnect, state=state):
        return False
    print("[GRIP] Closing gripper...")
    err = gripper_move(robot, GRIP_CLOSE_POS, reconnect=reconnect)
    print("[GRIP] Close retval:", err)
    if err == 0 and state is not None:
        state["gripper_closed"] = True
    time.sleep(0.3)
    return (err == 0)


def gripper_toggle(robot, reconnect=None, state=None):
    closed = None if state is None else state.get("gripper_closed", None)
    if closed is None:
        print("[GRIP] toggle: state unknown -> CLOSE first")
        return gripper_close(robot, reconnect=reconnect, state=state)
    return gripper_open(robot, reconnect=reconnect, state=state) if closed else gripper_close(robot, reconnect=reconnect, state=state)


# ------------------------------------------------------------
# ✅ MoveCart direct (원본 유지)
# ------------------------------------------------------------
def movecart_direct(robot, pose6, tool, user, vel_list, reconnect=None, label=""):
    pose6 = ensure_pose6(pose6)
    acc = 0.0
    ovl = 100.0

    for vv in vel_list:
        rtn = safe_call(
            robot.MoveCart,
            pose6, tool, user, float(vv), acc, ovl, -1.0, -1,
            retry=1,
            reconnect_cb=reconnect
        )
        if rtn == 0:
            if label:
                print(f"[MoveCart-OK] {label} vel={vv}")
            return 0
        if rtn == 112:
            if label:
                print(f"[MoveCart-112] {label} vel={vv} -> try next")
            continue
        if label:
            print(f"[MoveCart-FAIL] {label} vel={vv} err={rtn} -> try next")
    return 112


# ------------------------------------------------------------
# ✅ 9번 Smooth 시퀀스 (원본 유지)
# ------------------------------------------------------------
def run_smooth_sequence(robot, reconnect, tool, user, target_pose6, last_measure, vel_list):
    t0 = time.time()
    target_pose6 = ensure_pose6(target_pose6)
    z_hold = float(target_pose6[2]) + float(Z_HOLD_OFFSET_MM)

    (e1, cur_pose6), (e2, cur_joint6) = read_pose_joint(robot, reconnect=reconnect)
    if e1 != 0 or e2 != 0:
        print(f"[AUTO-STOP] ❌ 상태 읽기 실패 err_p={e1}, err_j={e2}")
        return False

    phase0_pose = target_pose6[:]
    phase0_pose[2] = z_hold

    if not has_solution(robot, phase0_pose, cur_joint6, reconnect=reconnect):
        print("[AUTO-STOP] ❌ phase0_pose IK 불가")
        print("  phase0_pose:", fmt_pose6(phase0_pose))
        return False

    print("\n[SMOOTH] 1) Phase0 MoveCart (XY 맞추고 Z hold까지 한번에)")
    rtn = movecart_direct(robot, phase0_pose, tool, user, vel_list, reconnect=reconnect, label="phase0")
    if rtn != 0:
        print("[AUTO-STOP] ❌ phase0 MoveCart 실패(112 포함)")
        return False

    print("\n[SMOOTH] 2) Rotate J6 (angle_deg 기반)")
    if last_measure is None:
        print("[SMOOTH-WARN] last_measure 없음(2번 미실행?) -> 회전 스킵")
    else:
        ang = float(last_measure.get("angle_deg", 0.0))
        delta = ANGLE_TO_J6_SIGN * ang
        delta = max(-J6_MAX_STEP_DEG, min(J6_MAX_STEP_DEG, delta))
        print(f"[SMOOTH] angle_deg={ang:+.3f} => delta(J6)={delta:+.3f}")
        # J6는 MoveJ 쓰는 함수가 위에 길어서, 너 원본 유지해도 됨(여긴 생략 가능)
        # (필요하면 너 코드의 move_j6_by_delta_deg 그대로 붙여서 사용)
        print("[SMOOTH-WARN] J6 회전 함수는 너 기존 코드 그대로 사용해줘 (현재 블록에서는 생략)")

    (e1, pose_after_rot), (e2, joint_after_rot) = read_pose_joint(robot, reconnect=reconnect)
    if e1 != 0 or e2 != 0:
        print(f"[AUTO-STOP] ❌ 회전 후 상태 읽기 실패 err_p={e1}, err_j={e2}")
        return False

    print("\n[SMOOTH] 3) Z Down MoveCart (Z만 한번에 내려감)")
    down_pose = pose_after_rot[:]
    down_pose[2] = float(target_pose6[2])

    if not has_solution(robot, down_pose, joint_after_rot, reconnect=reconnect):
        print("[AUTO-STOP] ❌ zdown_pose IK 불가")
        print("  down_pose:", fmt_pose6(down_pose))
        return False

    rtn = movecart_direct(robot, down_pose, tool, user, vel_list, reconnect=reconnect, label="zdown")
    if rtn != 0:
        print("[AUTO-STOP] ❌ Zdown MoveCart 실패(112 포함)")
        return False

    dt = time.time() - t0
    print(f"[SMOOTH] total_time={dt:.2f}s")
    return True


# ------------------------------------------------------------
# 메뉴
# ------------------------------------------------------------
def prompt_menu(step_scale):
    print("=======================================")
    print("무슨 기능을 할까요?")
    print("  1 : 현재 tcp_pose + joint(deg) 가져오기/저장")
    print("  2 : box_test_6 측정 실행/저장 (moveXYZ/angle)")
    print("  q : 종료")
    print("=======================================")
    return input("입력 (1/2/q) > ").strip()


def main():
    robot = connect_robot(ROBOT_IP)

    def reconnect():
        nonlocal robot
        try:
            robot.CloseRPC()
        except Exception:
            pass
        time.sleep(0.3)
        robot = connect_robot(ROBOT_IP)

    tool = 0
    user = 0

    step_scale = STEP_SCALE_DEFAULT

    last_tcp_pose = None
    last_measure = None

    # ✅ Live recorder 시작 (last_measure를 오버레이로 표시)
    recorder = None
    if ALWAYS_RECORD:
        def _get_last_measure():
            return last_measure
        recorder = LiveRecorder(_get_last_measure)
        recorder.start()
        print(f"[REC] ALWAYS_RECORD=ON  (dir={os.path.abspath(REC_DIR)})")
        print(f"[REC] overlay: moveXYZ(mm), angle(deg)\n")

    try:
        while True:
            cmd = prompt_menu(step_scale)

            if cmd == "1":
                print("\n[ACTION] GetActualTCPPose / Joint...")
                (e1, pose6), (e2, joint6) = read_pose_joint(robot, reconnect=reconnect)
                if e1 != 0 or e2 != 0:
                    print(f"[FAIL] err_p={e1}, err_j={e2}\n")
                    continue
                last_tcp_pose = pose6
                print("[OK] 현재 상태 저장 ✅")
                print("tcp_pose :", fmt_pose6(pose6))
                print("joint6   :", fmt_joint(joint6))
                print()

            elif cmd == "2":
                print("\n[ACTION] box_test_6 측정 시작...")

                # ✅ 측정할 때 카메라 충돌 방지: 상시녹화 pause -> 측정 -> resume
                if recorder is not None:
                    print("[REC] pause (measure_main uses camera)")
                    recorder.pause()
                    time.sleep(0.2)

                try:
                    res = measure_main()
                except Exception as e:
                    print(f"[ERROR] box_test_6 실행 예외: {e}\n")
                    res = None

                if recorder is not None:
                    time.sleep(0.2)
                    recorder.resume()
                    print("[REC] resume\n")

                if res is None:
                    print("[FAIL] 측정 실패(None)\n")
                    continue

                last_measure = res
                print("[OK] 측정 결과 저장 ✅")
                print(f"moveXYZ(mm) = ({float(res['move_x_mm']):+.1f}, {float(res['move_y_mm']):+.1f}, {float(res['move_z_mm']):+.1f})")
                print(f"angle(deg)  = {float(res.get('angle_deg', 0.0)):+.2f}\n")

            elif cmd.lower() == "q":
                print("\n[EXIT] 종료합니다.")
                break

            else:
                print("\n[WARN] 잘못된 입력입니다.\n")

    finally:
        if recorder is not None:
            recorder.stop()
        try:
            robot.CloseRPC()
        except Exception:
            pass


if __name__ == "__main__":
    main()

