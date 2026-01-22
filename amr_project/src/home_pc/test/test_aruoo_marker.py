#!/usr/bin/env python3
"""
ArUco 마커 감지 테스트 스크립트
RealSense D405 카메라로 ArUco 마커 실시간 감지
"""
import cv2
import numpy as np
import pyrealsense2 as rs
import sys

print("=" * 60)
print("ArUco 마커 감지 테스트")
print("=" * 60)

# ============================================================
# 설정
# ============================================================
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_SIZE_MM = 73.0  # 실제 마커 크기 (mm)

# ============================================================
# ArUco 초기화
# ============================================================
print("\n[1] ArUco 초기화 중...")
try:
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    
    # DetectorParameters 생성 (버전 호환)
    try:
        aruco_params = cv2.aruco.DetectorParameters()
        print("    ✅ DetectorParameters() 사용 (최신 버전)")
    except:
        aruco_params = cv2.aruco.DetectorParameters_create()
        print("    ✅ DetectorParameters_create() 사용 (구버전)")
    
    print(f"    ✅ Dictionary: DICT_4X4_50")
    print(f"    ✅ OpenCV version: {cv2.__version__}")
    
except Exception as e:
    print(f"    ❌ 초기화 실패: {e}")
    sys.exit(1)

# ============================================================
# RealSense 초기화
# ============================================================
print("\n[2] RealSense 카메라 초기화 중...")
try:
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 스트림 설정
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # 파이프라인 시작
    profile = pipeline.start(config)
    
    # Depth 센서 설정
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    print(f"    ✅ RealSense 연결됨")
    print(f"    ✅ Depth scale: {depth_scale}")
    
    # Align 객체 (Depth를 Color에 정렬)
    align = rs.align(rs.stream.color)
    
except Exception as e:
    print(f"    ❌ RealSense 초기화 실패: {e}")
    print("    💡 카메라가 연결되어 있는지 확인하세요")
    sys.exit(1)

# ============================================================
# 메인 루프
# ============================================================
print("\n[3] 마커 감지 시작")
print("=" * 60)
print("💡 사용법:")
print("   - 마커를 카메라 앞에 위치시키세요 (거리 60-80cm)")
print("   - 'q' 키를 누르면 종료")
print("   - 's' 키를 누르면 스크린샷 저장")
print("=" * 60)

frame_count = 0
detected_count = 0

try:
    while True:
        # 프레임 획득
        frames = pipeline.wait_for_frames(timeout_ms=1000)
        
        # Depth와 Color 정렬
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue
        
        # NumPy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # ArUco 마커 감지
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, 
            aruco_dict, 
            parameters=aruco_params
        )
        
        # 결과 이미지 복사
        display_image = color_image.copy()
        
        # 마커가 감지되었을 때
        if ids is not None and len(ids) > 0:
            detected_count += 1
            
            # 마커 그리기
            cv2.aruco.drawDetectedMarkers(display_image, corners, ids)
            
            # 각 마커 정보 표시
            for i, marker_id in enumerate(ids.flatten()):
                marker_corners = corners[i][0]
                
                # 중심점 계산
                center_x = int(np.mean(marker_corners[:, 0]))
                center_y = int(np.mean(marker_corners[:, 1]))
                
                # Depth 측정
                depth_value = depth_frame.get_distance(center_x, center_y)
                depth_mm = depth_value * 1000.0
                
                # 마커 크기 계산 (픽셀)
                corner_dists = []
                for j in range(4):
                    k = (j + 1) % 4
                    dist = np.linalg.norm(marker_corners[k] - marker_corners[j])
                    corner_dists.append(dist)
                marker_size_px = np.mean(corner_dists)
                
                # 정보 표시
                info_text = f"ID:{marker_id} Dist:{depth_mm:.0f}mm Size:{marker_size_px:.0f}px"
                cv2.putText(
                    display_image, 
                    info_text,
                    (center_x - 50, center_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
                
                # 콘솔 출력
                if frame_count % 30 == 0:  # 1초마다
                    print(f"[마커 감지] ID: {marker_id}, 거리: {depth_mm:.0f}mm, 크기: {marker_size_px:.0f}px")
        
        # 통계 표시
        detection_rate = (detected_count / max(1, frame_count)) * 100
        stats_text = f"Frame: {frame_count}  Detected: {detected_count}  Rate: {detection_rate:.1f}%"
        cv2.putText(
            display_image,
            stats_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )
        
        # rejected 마커 표시 (빨간색)
        if rejected is not None and len(rejected) > 0:
            cv2.aruco.drawDetectedMarkers(display_image, rejected, None, (0, 0, 255))
            rejected_text = f"Rejected: {len(rejected)}"
            cv2.putText(
                display_image,
                rejected_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
        
        # 화면 표시
        cv2.imshow('ArUco Detection Test', display_image)
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n종료합니다...")
            break
        elif key == ord('s'):
            filename = f"aruco_test_{frame_count}.png"
            cv2.imwrite(filename, display_image)
            print(f"스크린샷 저장: {filename}")
        
        frame_count += 1

except KeyboardInterrupt:
    print("\n\nCtrl+C로 종료")

except Exception as e:
    print(f"\n\n에러 발생: {e}")
    import traceback
    traceback.print_exc()

finally:
    # 정리
    print("\n정리 중...")
    try:
        pipeline.stop()
    except:
        pass
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    print(f"총 프레임: {frame_count}")
    print(f"마커 감지: {detected_count}")
    if frame_count > 0:
        print(f"감지율: {(detected_count/frame_count)*100:.1f}%")
    print("=" * 60)