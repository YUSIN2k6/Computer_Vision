import cv2
import os
from core.histogram_utils import equalize_histogram
from core.spatial_filters import apply_dog, apply_log

def main():
    # 1. Định nghĩa đường dẫn file
    # Đảm bảo bạn đã đổi tên 2 ảnh gốc trong thư mục inputs cho khớp với tên dưới đây,
    # hoặc sửa lại chuỗi string dưới đây cho khớp với tên file của bạn.
    input_img1 = "inputs/image_c5f202.png" 
    input_img2 = "inputs/image_c65321.png" 
    
    output_dir = "outputs/bt_26_08"

    # 2. Xử lý Ảnh 1: Cân bằng Histogram
    print("Đang xử lý Ảnh 1 (Cân bằng Histogram)...")
    try:
        img1_hist = equalize_histogram(input_img1)
        cv2.imwrite(os.path.join(output_dir, "anh1_sau_canbang.png"), img1_hist)
        print("-> Đã lưu: anh1_sau_canbang.png")
    except Exception as e:
        print(e)

    # 3. Xử lý Ảnh 2: Derivative of Gaussian (DoG)
    print("\Đang xử lý Ảnh 2 (DoG)...")
    try:
        img2_dog = apply_dog(input_img2)
        cv2.imwrite(os.path.join(output_dir, "anh2_sau_dog.png"), img2_dog)
        print("-> Đã lưu: anh2_sau_dog.png")
    except Exception as e:
        print(e)

    # 4. Xử lý Ảnh 2: Laplacian of Gaussian (LoG)
    print("\Đang xử lý Ảnh 2 (LoG)...")
    try:
        img2_log = apply_log(input_img2)
        cv2.imwrite(os.path.join(output_dir, "anh2_sau_log.png"), img2_log)
        print("-> Đã lưu: anh2_sau_log.png")
    except Exception as e:
        print(e)

    print("\nHOÀN THÀNH! Hãy kiểm tra thư mục outputs/bt_26_08 để lấy ảnh và đưa lên Canva.")

if __name__ == "__main__":
    main()