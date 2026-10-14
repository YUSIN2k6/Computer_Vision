import cv2
import os
import numpy as np

from core.gray import load_and_convert_to_gray
from core.features import (
    harris_corner_builtin,
    harris_corner_manual,
    harris_feature_matching,
)

if __name__ == "__main__":
    output_dir = "outputs/bt_9_9"
    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # CÂU 1: PHÁT HIỆN GÓC VỚI ẢNH CAPIBARA.JPG
    # ==========================================
    input_cau1 = "inputs/capibara.jpg"
    img_c1, gray_c1 = load_and_convert_to_gray(input_cau1)

    if img_c1 is not None:
        print("\n--- ĐANG XỬ LÝ CÂU 1 (capibara.jpg) ---")
        res_1a = harris_corner_builtin(img_c1, gray_c1)
        res_1b = harris_corner_manual(img_c1, gray_c1)

        out_path_1a = os.path.join(output_dir, "cau1a_harris.png")
        out_path_1b = os.path.join(output_dir, "cau1b_harris_manual.png")

        cv2.imwrite(out_path_1a, res_1a)
        cv2.imwrite(out_path_1b, res_1b)
        print(f"-> Đã lưu ảnh 1a tại: {out_path_1a}")
        print(f"-> Đã lưu ảnh 1b tại: {out_path_1b}")

        # Hiển thị kết quả Câu 1
        cv2.imshow("Cau 1a - Ham co san (Capibara)", res_1a)
        cv2.imshow("Cau 1b - Tinh thu cong (Capibara)", res_1b)

    # ==========================================
    # CÂU 2: MATCHING 2 ẢNH (cheems1 & cheems3)
    # ==========================================
    input_cau2_img1 = "inputs/cheems1.png"
    input_cau2_img2 = "inputs/cheems3.png"

    img1, gray1 = load_and_convert_to_gray(input_cau2_img1)
    img2, gray2 = load_and_convert_to_gray(input_cau2_img2)

    if img1 is not None and img2 is not None:
        print("\n--- ĐANG XỬ LÝ CÂU 2 (cheems1 & cheems3) ---")
        matched_result = harris_feature_matching(img1, gray1, img2, gray2)

        match_out_path = os.path.join(output_dir, "cau2_matching.png")
        cv2.imwrite(match_out_path, matched_result)
        print(f"-> Đã lưu ảnh kết quả Câu 2 tại: {match_out_path}")

        # Hiển thị kết quả Câu 2
        cv2.imshow("Cau 2: Harris Corner Matching", matched_result)

    # Đợi nhấn phím bất kỳ để đóng tất cả các cửa sổ ảnh
    cv2.waitKey(0)
    cv2.destroyAllWindows()