import cv2
import numpy as np

def apply_dog(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_path}")

    # Bước 1: Làm mờ Gaussian trên toàn bộ ảnh (giúp xử lý padding chuẩn xác)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Bước 2: Dùng đạo hàm bậc 1 Sobel X và Y
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Gộp biên X, Y và chuẩn hóa
    dog_edges = cv2.magnitude(sobel_x, sobel_y)
    dog_normalized = cv2.normalize(dog_edges, None, 0, 255, cv2.NORM_MINMAX)
    
    return dog_normalized.astype(np.uint8)

def apply_log(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_path}")

    # Bước 1: Làm mờ Gaussian trên toàn bộ ảnh
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Bước 2: Dùng đạo hàm bậc 2 Laplacian (Tổng ma trận lọc đảm bảo = 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # Bước 3: Lấy trị tuyệt đối và chuẩn hóa để viền sáng, nền đen tuyệt đối
    abs_lap = np.abs(laplacian)
    log_normalized = cv2.normalize(abs_lap, None, 0, 255, cv2.NORM_MINMAX)
    
    return log_normalized.astype(np.uint8)