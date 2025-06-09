import cv2
import numpy as np
import matplotlib.pyplot as plt

# Bước 1: Đọc hai hình ảnh ở chế độ màu
img1 = cv2.imread('o3.jpg', cv2.IMREAD_COLOR)  # Ảnh tấm vải lành
img2 = cv2.imread('t3.jpg', cv2.IMREAD_COLOR)  # Ảnh tấm vải rách

# Kiểm tra xem ảnh có được đọc thành công không
if img1 is None or img2 is None:
    print("Lỗi: Không thể đọc một trong hai ảnh!")
    exit()

# Đảm bảo kích thước ảnh giống nhau
if img1.shape[:2] != img2.shape[:2]:
    print("Lỗi: Kích thước ảnh không khớp!")
    exit()

# Làm mịn ảnh để giảm nhiễu
img1 = cv2.GaussianBlur(img1, (5, 5), 0)
img2 = cv2.GaussianBlur(img2, (5, 5), 0)

# Chuyển sang ảnh xám để áp dụng FFT
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Bước 2: Áp dụng FFT cho cả hai ảnh xám
f1 = np.fft.fft2(img1_gray)
f2 = np.fft.fft2(img2_gray)

# Dịch chuyển tần số thấp về trung tâm
fshift1 = np.fft.fftshift(f1)
fshift2 = np.fft.fftshift(f2)

# Tính magnitude spectrum
magnitude1 = 20 * np.log(np.abs(fshift1) + 1)
magnitude2 = 20 * np.log(np.abs(fshift2) + 1)

# Bước 3: Tính sự khác biệt trong miền tần số
diff_f = fshift1 - fshift2
diff_magnitude = 20 * np.log(np.abs(diff_f) + 1)

# Bước 4: Chuyển ngược về miền không gian
diff_ishift = np.fft.ifftshift(diff_f)
diff_spatial = np.fft.ifft2(diff_ishift)
diff_spatial = np.abs(diff_spatial)

# Chuẩn hóa để hiển thị
diff_spatial = cv2.normalize(diff_spatial, None, 0, 255, cv2.NORM_MINMAX)
diff_spatial = diff_spatial.astype(np.uint8)

# Bước 5: Tính phần trăm sai khác
total_pixels = img1.shape[0] * img1.shape[1]  # Tổng số pixel
diff_binary = cv2.threshold(diff_spatial,50 , 255, cv2.THRESH_BINARY)[1]  # Ngưỡng thấp để phát hiện mọi khác biệt nhỏ
diff_pixels = cv2.countNonZero(diff_binary)  # Số pixel khác biệt
difference_percentage = (diff_pixels / total_pixels) * 100

# Kết luận dựa trên phần trăm sai khác
threshold_percentage = 20  # Ngưỡng phần trăm (có thể điều chỉnh, ví dụ: 1% hoặc 2%)
if difference_percentage > threshold_percentage:
    conclusion = f"Ảnh bị lỗi! Phần trăm sai khác: {difference_percentage:.2f}% (ngưỡng: {threshold_percentage}%)"
else:
    conclusion = f"Ảnh không bị lỗi! Phần trăm sai khác: {difference_percentage:.2f}% (ngưỡng: {threshold_percentage}%)"
print(conclusion)

# Bước 6: Làm nổi bật vùng khác biệt
_, thresh = cv2.threshold(diff_spatial, 78, 255, cv2.THRESH_BINARY)  # Giảm ngưỡng để nhạy hơn
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Vẽ vùng khác biệt lên ảnh gốc màu
img1_color = img1.copy()  # Dùng ảnh màu gốc thay vì chuyển từ xám
for contour in contours:
    if cv2.contourArea(contour) > 35:
        x, y, w, h = cv2.boundingRect(contour)
        # Vẽ khung viền xanh lá cây
        cv2.rectangle(img1_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Làm nổi bật vùng bên trong bằng màu cam
        cv2.drawContours(img1_color, [contour], -1, (0, 165, 255), -1)

# Bước 7: Hiển thị kết quả
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title('Ảnh Gốc (Lành)')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title('Ảnh Bị Rách')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(magnitude1, cmap='gray')
plt.title('Phổ Tần Số (Ảnh Gốc)')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(magnitude2, cmap='gray')
plt.title('Phổ Tần Số (Ảnh Rách)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(diff_magnitude, cmap='gray')
plt.title('Sự Khác Biệt Trong Miền Tần Số')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB))
plt.title(f'Vùng Khác Biệt (Cam bên trong, Xanh Lá viền)\n{conclusion}')
plt.axis('off')

plt.tight_layout()
plt.show()

# Lưu ảnh kết quả nếu cần
cv2.imwrite('result.jpg', img1_color)