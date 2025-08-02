import cv2
import numpy as np

id = "0199"

# Đường dẫn ảnh
rgb_path = f'nyu_rgb_images/rgb_image_{id}.jpg'
depth_path = f'nyu_depth_images/depth_map_{id}.png'

# Đọc ảnh RGB
rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

# Đọc ảnh depth (grayscale)
depth_image = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

# Áp dụng colormap cho depth (chọn colormap phù hợp: JET, PLASMA, INFERNO, VIRIDIS,...)
colored_depth = cv2.applyColorMap(depth_image, cv2.COLORMAP_PLASMA)

# Hiển thị ảnh
cv2.imshow('RGB Image', cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
cv2.imshow('Depth Map (Color)', colored_depth)
cv2.waitKey(0)
cv2.destroyAllWindows()
