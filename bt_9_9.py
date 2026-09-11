import cv2
import os
import numpy as np

from core.gray import load_and_convert_to_gray
from core.features import harris_corner_builtin, harris_corner_manual

if __name__ == "__main__":
    input_image_path = "inputs/capibara.jpg" 
    output_dir = "outputs/bt_9_9"
    
    os.makedirs(output_dir, exist_ok=True)
    
    original_img, grayscale_img = load_and_convert_to_gray(input_image_path)
    
    if original_img is not None:
        print("Đang xử lý Câu 1a (Hàm có sẵn)...")
        res_1a = harris_corner_builtin(original_img, grayscale_img)
        
        print("Đang xử lý Câu 1b (Tính thủ công)...")
        res_1b = harris_corner_manual(original_img, grayscale_img)
        
        # Lưu riêng 2 ảnh vào 2 file khác nhau
        out_path_1a = os.path.join(output_dir, "cau1a_harris.png")
        out_path_1b = os.path.join(output_dir, "cau1b_harris_manual.png")
        
        cv2.imwrite(out_path_1a, res_1a)
        cv2.imwrite(out_path_1b, res_1b)
        
        print(f"-> Đã lưu ảnh 1a tại: {out_path_1a}")
        print(f"-> Đã lưu ảnh 1b tại: {out_path_1b}")

        # Hiển thị 2 cửa sổ riêng biệt để bạn dễ nhìn
        cv2.imshow("Cau 1a - Ham co san", res_1a)
        cv2.imshow("Cau 1b - Tinh thu cong", res_1b)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




# if __name__ == "__main__":
#     # --- CÂU 1: PHÁT HIỆN GÓC (Ảnh 1) ---
#     input_image_path1 = "inputs/cheems1.png" 
#     output_dir = "outputs/bt_9_9"
#     os.makedirs(output_dir, exist_ok=True)
    
#     img1, gray1 = load_and_convert_to_gray(input_image_path1)
    
#     if img1 is not None:
#         print("\n--- ĐANG XỬ LÝ CÂU 1 ---")
#         res_1a = harris_corner_builtin(img1, gray1)
#         res_1b = harris_corner_manual(img1, gray1)
        
#         cv2.imwrite(os.path.join(output_dir, "cau1a_harris.png"), res_1a)
#         cv2.imwrite(os.path.join(output_dir, "cau1b_harris_manual.png"), res_1b)
#         print("-> Đã lưu ảnh kết quả Câu 1a và 1b vào thư mục outputs.")

#     # --- CÂU 2: MATCHING 2 ẢNH (Ảnh 1 & Ảnh 2) ---
#     input_image_path2 = "inputs/cheems3.png"
#     img2, gray2 = load_and_convert_to_gray(input_image_path2)

#     if img1 is not None and img2 is not None:
#         print("\n--- ĐANG XỬ LÝ CÂU 2 ---")
#         matched_result = harris_feature_matching(img1, gray1, img2, gray2)
        
#         match_out_path = os.path.join(output_dir, "cau2_matching.png")
#         cv2.imwrite(match_out_path, matched_result)
#         print(f"-> Đã lưu ảnh kết quả Câu 2 tại: {match_out_path}")

#         # Hiển thị ảnh ghép nối
#         cv2.imshow("Cau 2: Harris Corner Matching", matched_result)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()