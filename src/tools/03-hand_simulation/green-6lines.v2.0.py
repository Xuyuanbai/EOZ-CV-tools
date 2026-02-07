"""
模拟人工法！！！
较1.0 减少了 def scan_radial_distance部分return的r来弥补停止时的蓝色像素比绿色外扩，也就是：
            if green_started and is_blue:
                return r - 1
"""

import cv2
import numpy as np


class GreenDiameterMeasurer:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"no image: {image_path}")

        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        self.green_colors = [
            np.array([68, 231, 210]),
            np.array([48, 255, 230])
        ]

        self.blue_colors = [
            np.array([92, 255, 252]),
            np.array([95, 255, 252]),
            np.array([98, 255, 252]),
            np.array([101, 245, 252])
        ]

    def is_green_pixel(self, pixel_hsv):
        for green_color in self.green_colors:
            if np.array_equal(pixel_hsv, green_color):
                return True
        return False

    def is_blue_pixel(self, pixel_hsv):
        for blue_color in self.blue_colors:
            if np.array_equal(pixel_hsv, blue_color):
                return True
        return False

    def create_color_mask(self):
        saturation = self.hsv[:, :, 1]
        color_mask = (saturation > 30).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        return color_mask

    def detect_outer_circle(self):
        color_mask = self.create_color_mask()
        edges = cv2.Canny(color_mask, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        circles = cv2.HoughCircles(
            edges, cv2.HOUGH_GRADIENT, dp=1,
            minDist=100, param1=50, param2=30,
            minRadius=50, maxRadius=1000
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            largest_circle = max(circles[0], key=lambda c: c[2])
            x, y, r = largest_circle
            return x, y, r
        else:
            raise ValueError("no big circle")

    def scan_radial_distance(self, center_x, center_y, angle_deg, max_radius):
        height, width = self.hsv.shape[:2]
        angle_rad = np.deg2rad(angle_deg)

        green_started = False

        for r in range(0, int(max_radius * 0.8)):
            x = int(center_x + r * np.cos(angle_rad))
            y = int(center_y + r * np.sin(angle_rad))

            if not (0 <= x < width and 0 <= y < height):
                continue

            pixel_hsv = self.hsv[y, x]
            is_green = self.is_green_pixel(pixel_hsv)
            is_blue = self.is_blue_pixel(pixel_hsv)

            if is_green:
                green_started = True

            if green_started and is_blue:
                return r - 1

        return -1

    def measure_diameters(self, show_image=True):

        # 检测圆心和半径
        center_x, center_y, outer_radius = self.detect_outer_circle()
        print(f"center: ({center_x}, {center_y}), r: {outer_radius}px")

        angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

        radial_distances = {}
        for angle in angles:
            distance = self.scan_radial_distance(center_x, center_y, angle, outer_radius)
            radial_distances[angle] = distance
            print(f"angle {angle:3d}°: dist = {distance:4d} px")

        diameter_pairs = [
            (0, 180),
            (30, 210),
            (60, 240),
            (90, 270),
            (120, 300),
            (150, 330)
        ]

        diameters = []
        valid_pairs = []

        for i, (angle1, angle2) in enumerate(diameter_pairs, 1):
            dist1 = radial_distances[angle1]
            dist2 = radial_distances[angle2]

            if dist1 > 0 and dist2 > 0:
                diameter = dist1 + dist2
                diameters.append(diameter)
                valid_pairs.append((angle1, angle2))
                print(f"直径 {i} ({angle1:3d}° - {angle2:3d}°): {diameter:4d} 像素")
            else:
                print(f"直径 {i} ({angle1:3d}° - {angle2:3d}°): error")

        if len(diameters) == 0:
            raise ValueError("no legal d found")

        average_diameter = np.mean(diameters)
        print(f"d avg: {average_diameter:.2f} px")
        print(f"success: {len(diameters)}/6 lines")

        # debug图形
        result_image = None
        if show_image:
            result_image = self.image.copy()

            for angle1, angle2 in valid_pairs:
                dist1 = radial_distances[angle1]
                dist2 = radial_distances[angle2]

                # 计算端点
                angle1_rad = np.deg2rad(angle1)
                angle2_rad = np.deg2rad(angle2)

                x1 = int(center_x + dist1 * np.cos(angle1_rad))
                y1 = int(center_y + dist1 * np.sin(angle1_rad))

                x2 = int(center_x + dist2 * np.cos(angle2_rad))
                y2 = int(center_y + dist2 * np.sin(angle2_rad))

                cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 0), 2)

            # 绘制圆心
            cv2.circle(result_image, (center_x, center_y), 5, (255, 0, 0), -1)

            # 显示图像
            #print(f"图像形状: {result_image.shape}, 数据类型: {result_image.dtype}")
            #cv2.imshow("result", result_image)
            cv2.imwrite("result v2.0.png", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return diameters, average_diameter, result_image


if __name__ == "__main__":
    image_path = "../../../tests/002.png"

    try:
        measurer = GreenDiameterMeasurer(image_path)
        diameters, avg_diameter, result_img = measurer.measure_diameters(show_image=True)

        print(f"\nresult:")
        print(f"6条直径: {diameters}")
        print(f"平均直径: {avg_diameter:.2f} 像素")

    except Exception as e:
        print(f"error: {e}")
        import traceback
        traceback.print_exc()