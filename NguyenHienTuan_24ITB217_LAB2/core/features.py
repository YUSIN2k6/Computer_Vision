import cv2
import numpy as np

def harris_corner_builtin(original_img, gray_img):
    """
    Câu 1a: Phát hiện góc Harris sử dụng hàm có sẵn cv2.cornerHarris
    """
    # Tạo một bản sao của ảnh gốc để vẽ đè lên (không làm hỏng ảnh gốc)
    result_img = original_img.copy()

    # 1. Chuyển đổi ảnh xám sang float32
    gray_float = np.float32(gray_img)

    # 2. Gọi hàm Harris có sẵn của OpenCV
    # blockSize=2, ksize=3, k=0.04
    dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)

    # 3. (Tùy chọn) Phình to các điểm ảnh góc để dễ nhìn hơn bằng mắt thường
    dst = cv2.dilate(dst, None)

    # 4. Lọc ngưỡng: Chọn các điểm có phản hồi R > 1% so với điểm R lớn nhất
    threshold = 0.01 * dst.max()
    
    # 5. Tô màu Đỏ (BGR: 0, 0, 255) cho các điểm thỏa mãn ngưỡng trên ảnh màu
    result_img[dst > threshold] = [0, 0, 255]

    return result_img

def harris_corner_manual(original_img, gray_img, k=0.04, threshold_ratio=0.01):
    """
    Câu 1b: Phát hiện góc Harris tính toán thủ công từng bước theo công thức.
    """
    result_img = original_img.copy()
    
    # 1. Tính đạo hàm bậc 1 theo hướng x và y (Ix, Iy) bằng bộ lọc Sobel
    # cv2.CV_64F giúp giữ độ chính xác của số thực có dấu (tránh bị tràn số)
    Ix = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    
    # 2. Tính các tích của đạo hàm (Ix^2, Iy^2, Ixy)
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy
    
    # 3. Làm trơn bằng Gaussian (Window function) để tạo S_x2, S_y2, S_xy
    Sxx = cv2.GaussianBlur(Ixx, (3, 3), 0)
    Syy = cv2.GaussianBlur(Iyy, (3, 3), 0)
    Sxy = cv2.GaussianBlur(Ixy, (3, 3), 0)
    
    # 4 & 5. Tính ma trận M và giá trị phản hồi R = det(M) - k*(trace(M))^2
    det_M = (Sxx * Syy) - (Sxy ** 2)
    trace_M = Sxx + Syy
    R = det_M - k * (trace_M ** 2)
    
    # 6. Lọc ngưỡng (Thresholding)
    threshold = threshold_ratio * R.max()
    
    # Tô màu Xanh lá cây (BGR: 0, 255, 0) cho các điểm góc tính thủ công
    result_img[R > threshold] = [0, 255, 0]
    
    return result_img

def harris_feature_matching(img1, gray1, img2, gray2):
    """
    Câu 2: Áp dụng Harris corner detection để matching 2 ảnh ở 2 góc nhìn khác nhau.
    """
    # 1. Tìm góc Harris cho ảnh 1 và tạo danh sách KeyPoint
    dst1 = cv2.cornerHarris(np.float32(gray1), 2, 3, 0.04)
    thresh1 = 0.01 * dst1.max()
    # Lấy tọa độ (y, x) của các điểm thỏa mãn ngưỡng
    kp1_coords = np.argwhere(dst1 > thresh1)
    # Chuyển đổi sang định dạng KeyPoint của OpenCV (x, y, size)
    kp1 = [cv2.KeyPoint(float(x), float(y), 10) for y, x in kp1_coords]

    # 2. Tìm góc Harris cho ảnh 2 và tạo danh sách KeyPoint
    dst2 = cv2.cornerHarris(np.float32(gray2), 2, 3, 0.04)
    thresh2 = 0.01 * dst2.max()
    kp2_coords = np.argwhere(dst2 > thresh2)
    kp2 = [cv2.KeyPoint(float(x), float(y), 10) for y, x in kp2_coords]

    # 3. Trích xuất vector đặc trưng (Descriptor) tại các góc Harris bằng SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.compute(gray1, kp1)
    kp2, des2 = sift.compute(gray2, kp2)

    # 4. Tìm điểm giống nhau (Matching) giữa 2 bộ descriptor
    # Sử dụng Brute-Force Matcher với khoảng cách L2
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sắp xếp các điểm khớp theo khoảng cách (khoảng cách nhỏ = độ giống nhau cao)
    matches = sorted(matches, key=lambda x: x.distance)

    # 5. Vẽ 50 đường nối tốt nhất lên cùng một khung hình
    matched_img = cv2.drawMatches(
        img1, kp1, img2, kp2, matches[:300], None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return matched_img