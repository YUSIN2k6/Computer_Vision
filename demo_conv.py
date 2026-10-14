import numpy as np
import matplotlib.pyplot as plt
import os

def conv2d_demo(image, kernel):
    # Kích thước kernel
    k_H, k_W = kernel.shape
    # Kích thước ảnh đầu ra (không có padding, stride = 1)
    out_H = image.shape[0] - k_H + 1
    out_W = image.shape[1] - k_W + 1
    output = np.zeros((out_H, out_W))
    
    # Trượt kernel qua ảnh
    for i in range(out_H):
        for j in range(out_W):
            region = image[i:i+k_H, j:j+k_W]
            output[i, j] = np.sum(region * kernel)
    return output

# 1. Tạo ma trận ảnh gốc 5x5 (Nửa trái tối - 0, nửa phải sáng - 10)
image = np.array([
    [0, 0, 0, 10, 10],
    [0, 0, 0, 10, 10],
    [0, 0, 0, 10, 10],
    [0, 0, 0, 10, 10],
    [0, 0, 0, 10, 10]
])

# 2. Kernel 3x3 (Bộ lọc phát hiện ranh giới/cạnh dọc)
kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

# 3. Tính toán Feature Map (Ma trận kết quả)
output_image = conv2d_demo(image, kernel)

# ==========================================
# TRỰC QUAN HÓA VÀ LƯU ẢNH (MATPLOTLIB)
# ==========================================
def plot_matrix(ax, matrix, title, cmap='Blues'):
    """Hàm vẽ ma trận thành các ô vuông màu kèm số"""
    cax = ax.matshow(matrix, cmap=cmap)
    ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
    
    # In con số lên từng ô màu để minh họa cách tính
    for (i, j), val in np.ndenumerate(matrix):
        # Đổi màu chữ (trắng/đen) sao cho tương phản với nền
        text_color = "white" if val > np.max(matrix)/2 else "black"
        ax.text(j, i, f'{val}', ha='center', va='center', color=text_color, fontsize=14)
    
    # Ẩn các trục tọa độ x, y đi cho giống bức ảnh
    ax.set_xticks([])
    ax.set_yticks([])

# Khởi tạo khung vẽ có 3 cột
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Đổ màu ma trận (Sử dụng các dải màu khác nhau để phân biệt)
plot_matrix(axes[0], image, "1. Anh Goc (5x5)", cmap='Greys')
plot_matrix(axes[1], kernel, "2. Kernel (3x3)", cmap='coolwarm')
plot_matrix(axes[2], output_image, "3. Feature Map (3x3)", cmap='Reds')

plt.tight_layout()

# LƯU ẢNH: Tự động lưu thành file PNG ngay trong thư mục CV_Workspace
save_path = 'convolution_demo_result.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Đã lưu thành công tấm ảnh minh họa tại: {os.path.abspath(save_path)}")

# Hiển thị cửa sổ hình ảnh lên màn hình (Dùng lúc chạy demo trực tiếp)
plt.show()