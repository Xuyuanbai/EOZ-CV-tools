"""
代码版本是src/tools/03-hand_simulation/green-6lines.v4.0.py，r目前为-1.
写给自己：暂时未删除图像输出。运行batch前记得删除！！！ 已删除(为了方便后续引用不添加新参数，直接注释掉了，放弃了用于debug的if条件语句)
较原版增加了报错时的提醒
===============
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
        blurred = cv2.medianBlur(self.image, 7)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2Lab)
        a_chan = lab[:, :, 1].astype(float)
        b_chan = lab[:, :, 2].astype(float)
        dist = np.sqrt((a_chan - 128) ** 2 + (b_chan - 128) ** 2)
        dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        _, thresh = cv2.threshold(dist, 35, 255, cv2.THRESH_BINARY)
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)

        # 距离变换找圆心
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)

        center_x, center_y = max_loc
        radius = max_val
        x, y, r = center_x, center_y, radius
        return x, y, r

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
                return r

        return -1

    def measure_diameters(self, show_image=True):

        # 检测圆心和半径
        center_x, center_y, outer_radius = self.detect_outer_circle()
        #print(f"center: ({center_x}, {center_y}), r: {outer_radius}px")

        angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

        radial_distances = {}
        for angle in angles:
            distance = self.scan_radial_distance(center_x, center_y, angle, outer_radius)
            radial_distances[angle] = distance
            #print(f"angle {angle:3d}°: dist = {distance:4d} px")

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
                #print(f"直径 {i} ({angle1:3d}° - {angle2:3d}°): {diameter:4d} 像素")
            else:
                print(f" 直径 {i} ({angle1:3d}° - {angle2:3d}°): error {dist1} & {dist2}")

        if len(diameters) == 0:
            raise ValueError("no legal d found")

        average_diameter = np.mean(diameters)
        #print(f"d avg: {average_diameter:.2f} px")
        #print(f"success: {len(diameters)}/6 lines")
        '''
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
            cv2.imwrite("tools/03-hand_simulation/result v2.0.png", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        '''
        return average_diameter #, result_image


if __name__ == "__main__":
    image_path = "../tests/1确定圆5.6mm.png"

    try:
        measurer = GreenDiameterMeasurer(image_path)
        avg_diameter = measurer.measure_diameters(show_image=True)

        print(f"\nresult:")
        #print(f"6条直径: {diameters}")
        print(f"平均直径: {avg_diameter:.2f} 像素")

    except Exception as e:
        print(f"error: {e}")
        import traceback
        traceback.print_exc()