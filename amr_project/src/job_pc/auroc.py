
#!/usr/bin/env python3
"""
ArUco 마커 생성 + A4 합본(6개/장) 출력 스크립트

- Table_1: ID 1 마커 PNG 생성
- Table_2: ID 2 마커 PNG 생성
- A4 한 장에 2열×3행 = 6개 배치한 출력물 생성

출력물:
  aruco_markers/Table_1_ID_1.png
  aruco_markers/Table_2_ID_2.png

  reportlab 설치됨 -> PDF:
    aruco_markers/A4_Table_1_ID_1_x6.pdf
    aruco_markers/A4_Table_2_ID_2_x6.pdf
    aruco_markers/A4_MIX_ID1_ID2_x6.pdf

  reportlab 미설치 -> PNG(300DPI):
    aruco_markers/A4_Table_1_ID_1_x6_300dpi.png
    aruco_markers/A4_Table_2_ID_2_x6_300dpi.png
    aruco_markers/A4_MIX_ID1_ID2_x6_300dpi.png

✅ 중요:
- "검은색 마커 본체" 크기를 MARKER_SIZE_MM(기본 80mm)로 맞춰 출력하도록 자동 스케일링
- 인쇄할 때 "실제 크기(100%)" / "페이지에 맞춤" OFF
"""

import os
import cv2
import numpy as np

# -----------------------------
# 설정
# -----------------------------
ARUCO_DICT = cv2.aruco.DICT_4X4_50

MARKER_SIZE_PX = 500  # 마커 본체(검정 바깥사각 포함) 픽셀
BORDER_PX = 50        # 흰 테두리(잘림 방지) 픽셀

MARKER_SIZE_MM = 80   # ✅ 실제 인쇄 시 "검은 마커 본체" 크기(mm)

MARKER_IDS = {
    "Table_1": 1,
    "Table_2": 2,
}

OUTPUT_DIR = "aruco_markers"


# -----------------------------
# 유틸
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def generate_marker_image(aruco_dict, marker_id: int, marker_size_px: int) -> np.ndarray:
    """
    OpenCV 버전 호환: generateImageMarker / drawMarker
    반환: 1채널 uint8(0~255) 이미지
    """
    if hasattr(cv2.aruco, "generateImageMarker"):
        img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px)
    else:
        img = np.zeros((marker_size_px, marker_size_px), dtype=np.uint8)
        cv2.aruco.drawMarker(aruco_dict, marker_id, marker_size_px, img, 1)
    return img


def save_marker_png(table_name: str, marker_id: int, aruco_dict) -> str:
    """
    마커 생성(500px) + 흰 테두리 추가(50px) + PNG 저장
    """
    marker_img = generate_marker_image(aruco_dict, marker_id, MARKER_SIZE_PX)

    bordered_img = cv2.copyMakeBorder(
        marker_img,
        BORDER_PX, BORDER_PX, BORDER_PX, BORDER_PX,
        cv2.BORDER_CONSTANT,
        value=255
    )

    out_path = os.path.join(OUTPUT_DIR, f"{table_name}_ID_{marker_id}.png")
    ok = cv2.imwrite(out_path, bordered_img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {out_path}")
    return out_path


def calc_total_mm_for_print() -> float:
    """
    지금 저장되는 PNG는 "흰테두리 포함" 전체가 (500+100)=600px.
    출력 배치 시 이 전체 이미지 크기를 total_mm로 맞추면,
    검은 본체(500px)가 MARKER_SIZE_MM이 된다.

    total_mm = MARKER_SIZE_MM * (total_px / marker_px)
             = 80 * (600/500) = 96mm
    """
    total_px = MARKER_SIZE_PX + 2 * BORDER_PX
    return float(MARKER_SIZE_MM) * (float(total_px) / float(MARKER_SIZE_PX))


# -----------------------------
# A4 합본 (PDF: reportlab)
# -----------------------------
def make_a4_pdf(pdf_path: str, image_paths: list, copies: int = 6):
    """
    reportlab 필요
    A4 한 장에 2x3 = 6개 배치 PDF 생성
    """
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader

    page_w_pt, page_h_pt = A4

    total_mm = calc_total_mm_for_print()
    img_w_pt = total_mm * mm
    img_h_pt = total_mm * mm

    cols, rows = 2, 3
    n = min(int(copies), cols * rows)

    # 남는 여백을 균등 분배
    # (reportlab 좌표는 왼쪽 아래가 원점)
    a4_w_mm, a4_h_mm = 210.0, 297.0
    used_w_mm = cols * total_mm
    used_h_mm = rows * total_mm
    margin_x_mm = max(0.0, (a4_w_mm - used_w_mm) / 2.0)
    margin_y_mm = max(0.0, (a4_h_mm - used_h_mm) / 2.0)

    margin_x_pt = margin_x_mm * mm
    margin_y_pt = margin_y_mm * mm

    c = canvas.Canvas(pdf_path, pagesize=A4)
    readers = [ImageReader(p) for p in image_paths]

    idx = 0
    for r in range(rows):
        for col in range(cols):
            if idx >= n:
                break

            x = margin_x_pt + col * img_w_pt
            # 위에서 아래로 채우기 위해 y 계산
            y = page_h_pt - margin_y_pt - (r + 1) * img_h_pt

            reader = readers[idx % len(readers)]
            c.drawImage(reader, x, y, width=img_w_pt, height=img_h_pt,
                        preserveAspectRatio=True, mask='auto')
            idx += 1

        if idx >= n:
            break

    c.showPage()
    c.save()


# -----------------------------
# A4 합본 (PNG: reportlab 없이)
# -----------------------------
def make_a4_png(out_path: str, image_paths: list, copies: int = 6, dpi: int = 300):
    """
    reportlab 없이 A4 한 장짜리 인쇄용 PNG 생성 (기본 300DPI)
    - A4: 210x297mm
    - 2x3 배치
    - "검은 마커 본체" 80mm가 되도록 스케일 적용
    """
    A4_W_MM, A4_H_MM = 210.0, 297.0

    a4_w_px = int(round(A4_W_MM * dpi / 25.4))
    a4_h_px = int(round(A4_H_MM * dpi / 25.4))

    total_mm = calc_total_mm_for_print()
    tile_px = int(round(total_mm * dpi / 25.4))  # 정사각 타일

    cols, rows = 2, 3
    n = min(int(copies), cols * rows)

    used_w = cols * tile_px
    used_h = rows * tile_px
    margin_x = max(0, (a4_w_px - used_w) // 2)
    margin_y = max(0, (a4_h_px - used_h) // 2)

    canvas_img = np.full((a4_h_px, a4_w_px, 3), 255, dtype=np.uint8)

    # 이미지 로드 + 리사이즈 캐시
    loaded = []
    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {p}")
        img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img3 = cv2.resize(img3, (tile_px, tile_px), interpolation=cv2.INTER_NEAREST)
        loaded.append(img3)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n:
                break
            x = margin_x + c * tile_px
            y = margin_y + r * tile_px
            tile = loaded[idx % len(loaded)]
            canvas_img[y:y + tile_px, x:x + tile_px] = tile
            idx += 1
        if idx >= n:
            break

    ok = cv2.imwrite(out_path, canvas_img)
    if not ok:
        raise RuntimeError(f"Failed to write A4 PNG: {out_path}")


def try_make_a4_outputs(table_pngs: dict):
    """
    reportlab 있으면 PDF, 없으면 PNG로 자동 생성
    """
    # 출력물 이름
    pdf1 = os.path.join(OUTPUT_DIR, "A4_Table_1_ID_1_x6.pdf")
    pdf2 = os.path.join(OUTPUT_DIR, "A4_Table_2_ID_2_x6.pdf")
    pdfm = os.path.join(OUTPUT_DIR, "A4_MIX_ID1_ID2_x6.pdf")

    png1 = os.path.join(OUTPUT_DIR, "A4_Table_1_ID_1_x6_300dpi.png")
    png2 = os.path.join(OUTPUT_DIR, "A4_Table_2_ID_2_x6_300dpi.png")
    pngm = os.path.join(OUTPUT_DIR, "A4_MIX_ID1_ID2_x6_300dpi.png")

    try:
        # reportlab 존재 확인 + PDF 생성
        import reportlab  # noqa: F401
        make_a4_pdf(pdf1, [table_pngs["Table_1"]], copies=6)
        make_a4_pdf(pdf2, [table_pngs["Table_2"]], copies=6)
        make_a4_pdf(pdfm, [table_pngs["Table_1"], table_pngs["Table_2"]], copies=6)
        print("\n✅ reportlab 감지됨 → PDF 생성 완료")
        print(f" - {pdf1}")
        print(f" - {pdf2}")
        print(f" - {pdfm}")
    except Exception as e:
        # reportlab이 없거나 PDF 생성 실패 → PNG로 fallback
        print("\n⚠️ PDF 생성 불가 (reportlab 미설치 or 오류) → PNG로 생성합니다.")
        print(f"   이유: {type(e).__name__}: {e}")

        make_a4_png(png1, [table_pngs["Table_1"]], copies=6, dpi=300)
        make_a4_png(png2, [table_pngs["Table_2"]], copies=6, dpi=300)
        make_a4_png(pngm, [table_pngs["Table_1"], table_pngs["Table_2"]], copies=6, dpi=300)

        print("\n✅ PNG(300DPI) 생성 완료")
        print(f" - {png1}")
        print(f" - {png2}")
        print(f" - {pngm}")


# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dir(OUTPUT_DIR)

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    print("=" * 60)
    print("ArUco 마커 PNG 생성 중...")
    print("=" * 60)

    table_pngs = {}
    for table_name, marker_id in MARKER_IDS.items():
        path = save_marker_png(table_name, marker_id, aruco_dict)
        table_pngs[table_name] = path

        total_px = MARKER_SIZE_PX + 2 * BORDER_PX
        print(f"\n✅ {table_name}")
        print(f"   - Marker ID: {marker_id}")
        print(f"   - 파일: {path}")
        print(f"   - 마커 본체: {MARKER_SIZE_PX}x{MARKER_SIZE_PX}px")
        print(f"   - 흰 테두리: {BORDER_PX}px (총 {total_px}x{total_px}px)")

    print("\n" + "=" * 60)
    print("A4 합본 출력물 생성 중... (2열 x 3행 = 6개/장)")
    print("=" * 60)

    try_make_a4_outputs(table_pngs)

    total_mm = calc_total_mm_for_print()

    print("\n" + "=" * 60)
    print("📐 인쇄 가이드(중요)")
    print("=" * 60)
    print("1) 출력물(PDF/PNG)을 '실제 크기(100%)'로 인쇄하세요. (페이지에 맞춤 ❌)")
    print(f"2) 이 출력물은 '검은 마커 본체'가 {MARKER_SIZE_MM}mm가 되도록 스케일되어 있습니다.")
    print(f"   - 참고: 흰 테두리 포함 전체 타일은 약 {total_mm:.1f}mm 입니다.")
    print("3) 테이블에는 마커를 2~4개 정도 모서리에 분산 부착 추천")
    print("=" * 60)
    print(f"✅ 완료! {OUTPUT_DIR}/ 폴더 확인")


if __name__ == "__main__":
    main()
