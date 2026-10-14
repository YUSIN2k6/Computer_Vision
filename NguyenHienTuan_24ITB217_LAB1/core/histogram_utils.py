import cv2

def equalize_histogram(image_path):
    """
    Đọc ảnh xám và thực hiện cân bằng Histogram.
    """
    # Đọc ảnh ở chế độ xám (grayscale)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
    
    # Thực hiện cân bằng Histogram
    equ_img = cv2.equalizeHist(img)
    return equ_img