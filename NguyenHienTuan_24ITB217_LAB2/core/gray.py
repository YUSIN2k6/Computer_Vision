import cv2
import os

def load_and_convert_to_gray(image_path):
    """
    Hàm đọc ảnh từ đường dẫn và chuyển đổi sang ảnh xám.
    
    Tham số:
        image_path (str): Đường dẫn đến file ảnh (ví dụ: 'inputs/image_9963c7.png')
        
    Trả về:
        img (numpy.ndarray): Ảnh màu gốc.
        gray_img (numpy.ndarray): Ảnh đã được chuyển sang thang độ xám (grayscale).
    """
    # 1. Đọc ảnh gốc bằng OpenCV
    img = cv2.imread(image_path)
    
    # Kiểm tra xem ảnh có đọc thành công không (tránh lỗi sai đường dẫn)
    if img is None:
        print(f"Lỗi: Không tìm thấy hoặc không thể đọc ảnh tại đường dẫn '{image_path}'")
        return None, None

    # 2. Chuyển đổi ảnh từ hệ màu BGR (mặc định của OpenCV) sang màu xám (GRAY)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print(f"Đã chuyển đổi ảnh '{image_path}' sang màu xám thành công!")
    return img, gray_img