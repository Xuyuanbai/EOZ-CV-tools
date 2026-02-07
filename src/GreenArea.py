"""
此脚本中的方法来自src/tools/02-find_green_area/tools-GreenArea_v2.0.py
注释掉了用于debug的图像输出，只输出绿色圈内的像素总数；
调试：同时改变了经向扫描的像素数，目前为r = r -1或r，详见源代码的开头注释。 157行
      另：int（）或round（）


===划掉===
此脚本中的方法来自src/tools/02-find_green_area/tools-GreenArea_v1.0.py
注释掉了用于debug的图像输出，只输出绿色圈内的像素总数
===/划掉===



为了使过程更加工程化和模块化，基于既往的汇总脚本../00-old_script/core.py进行拆分，获取到如下提取绿色轮廓的核心代码：
input：path
process：定义了一个class，用来识别绿色边界（核心算法）
    class GreenContourExtractor:
        #启动
        def __init__(self, image_path)

        #定义和找出绿色和蓝色像素值
        def is_green_pixel(self, pixel_hsv)
        def is_blue_pixel(self, pixel_hsv)

        #检测外圆并确定中心（绿色中心会包含在内）
        def create_color_mask(self, debug=False)
        def detect_outer_circle(self, debug=False)

        #基于中心的绿色轮廓提取（核心！！！）：1.提前限定绿色位于0.8*大圆半径内 2.利用for循环基于中心进行经线扫描 3.去除离群+插值补充缺失 4.填充绿色轮廓并获取mask
        def extract_green_contour(self, center_x, center_y, outer_radius, debug=False)

        #对以上获取的对象汇总，进行最终计算
        def extract(self, debug=True, save_path=None)
output：green_mask图片、像素总数
"""


import cv2
import numpy as np
from scipy.interpolate import interp1d


class GreenContourExtractor:
    def __init__(self, image_path):

        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"无法读取图像: {image_path}")

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

    def create_color_mask(self, debug=False):
        saturation = self.hsv[:, :, 1]
        color_mask = (saturation > 30).astype(np.uint8) * 255

        kernel = np.ones((5, 5), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

        if debug:
            color_pixel_count = np.sum(color_mask > 0)
            print(f"彩色区域像素数: {color_pixel_count}")

        return color_mask

    def detect_outer_circle(self, debug=False):
        color_mask = self.create_color_mask(debug=debug)
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

            if debug:
                print(f"外圆检测 - 圆心: ({x}, {y}), 半径: {r}像素")

            return x, y, r
        else:
            raise ValueError("未能检测到外圆边界")

    def extract_green_contour(self, center_x, center_y, outer_radius, debug=False):
        height, width = self.hsv.shape[:2]
        contour_mask = np.zeros((height, width), dtype=np.uint8)

        max_scan_radius = int(outer_radius * 0.8)

        if debug:
            print(f"最大扫描半径: {max_scan_radius}像素")

        num_angles = 360
        angles = np.linspace(0, 2 * np.pi, num_angles)
        boundary_radii = []

        green_found_count = 0
        blue_found_count = 0

        for angle in angles:
            green_end_radius = -1
            green_started = False

            for r in range(0, max_scan_radius):
                x = int(center_x + r * np.cos(angle))
                y = int(center_y + r * np.sin(angle))

                if not (0 <= x < width and 0 <= y < height):
                    continue

                pixel_hsv = self.hsv[y, x]

                is_green = self.is_green_pixel(pixel_hsv)
                is_blue = self.is_blue_pixel(pixel_hsv)

                if is_green:
                    green_found_count += 1
                if is_blue:
                    blue_found_count += 1

                if is_green:
                    green_started = True

                if green_started and is_blue:
                    green_end_radius = r - 0.5
                    break

            boundary_radii.append(green_end_radius)

        if debug:
            print(f"扫描过程中找到的绿色像素总数: {green_found_count}")
            print(f"扫描过程中找到的蓝色像素总数: {blue_found_count}")

        boundary_radii = np.array(boundary_radii)
        valid_radii = boundary_radii[boundary_radii != -1]

        if len(valid_radii) < 3:
            if debug:
                print(f"\n[警告] 有效边界点太少: {len(valid_radii)}")
                print(f"绿色像素数: {green_found_count}, 蓝色像素数: {blue_found_count}")
            raise ValueError(f"有效边界点太少({len(valid_radii)}),无法分析")

        # 剔除离群值
        q1 = np.percentile(valid_radii, 25)
        q3 = np.percentile(valid_radii, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (valid_radii < lower_bound) | (valid_radii > upper_bound)
        outlier_count = np.sum(outlier_mask)

        if debug:
            print(f"统计信息:")
            print(f"  有效半径数量: {len(valid_radii)}")
            print(f"  中位数: {np.median(valid_radii):.2f}像素")
            print(f"  正常范围: [{lower_bound:.2f}, {upper_bound:.2f}]像素")
            print(f"  离群值数量: {outlier_count}")

        valid_indices = np.where(boundary_radii != -1)[0]
        for i, idx in enumerate(valid_indices):
            if outlier_mask[i]:
                boundary_radii[idx] = -1

        # 插值
        missing_count = np.sum(boundary_radii == -1)

        if missing_count > 0:
            normal_indices = np.where(boundary_radii != -1)[0]
            normal_radii = boundary_radii[normal_indices]

            if len(normal_indices) < 3:
                raise ValueError("正常值太少,无法插值")

            # 环形插值
            extended_indices = np.concatenate([
                [normal_indices[0] - num_angles],
                normal_indices,
                [normal_indices[-1] + num_angles]
            ])
            extended_radii = np.concatenate([
                [normal_radii[0]],
                normal_radii,
                [normal_radii[-1]]
            ])

            interp_func = interp1d(extended_indices, extended_radii,
                                   kind='cubic', fill_value='extrapolate')

            all_indices = np.arange(num_angles)
            boundary_radii = interp_func(all_indices)
            boundary_radii = np.clip(boundary_radii, 0, max_scan_radius)

        # 生成边界
        boundary_points = []
        for i, angle in enumerate(angles):
            r = boundary_radii[i]
            x = int(center_x + r * np.cos(angle))
            y = int(center_y + r * np.sin(angle))
            x = np.clip(x, 0, width - 1)
            y = np.clip(y, 0, height - 1)
            boundary_points.append((x, y))

        # 填充
        contour_points = np.array(boundary_points, dtype=np.int32)
        cv2.fillPoly(contour_mask, [contour_points], 255)

        if debug:
            print(f"边界点数: {len(boundary_points)}")
            print(f"轮廓内像素数: {np.sum(contour_mask > 0)}")
            if missing_count > 0:
                print(f"已插值修复 {missing_count} 个角度(含{outlier_count}个离群值)")

        return contour_mask

    def extract(self, debug=True, save_path=None):
        """
        执行完整的绿色轮廓提取流程

        Args:
            debug: 是否输出调试信息
            save_path: 保存掩码图的路径(可选)

        Returns:
            绿色轮廓掩码图
        """
        #下面这些输print可以被注释掉
        '''
        print("=" * 60)
        #print("开始提取绿色轮廓线...")
        #print("=" * 60)

        # 1. 检测外圆
        print("\n[步骤1] 检测外圆边界...")
        '''
        center_x, center_y, outer_radius = self.detect_outer_circle(debug=debug)

        # 2. 提取绿色轮廓
        '''
        print("\n[步骤2] 提取绿色轮廓边界(精确颜色匹配)...")
        print("  - 绿色: HSV(68,231,210) 和 HSV(48,255,230)")
        print("  - 蓝色: 4个精确采样值")
        print("  - 边界检测: 绿色消失→蓝色出现")
        '''
        contour_mask = self.extract_green_contour(
            center_x, center_y, outer_radius, debug=debug
        )

        # 3. 保存结果，注意这里的if save_path也被注释掉了，如果需要导出掩码图，需要修改这里的代码。

        if save_path:
            cv2.imwrite(save_path, contour_mask)
            #print(f"\n掩码图已保存到: {save_path}")
        '''
        print("\n" + "=" * 60)
        print("绿色轮廓提取完成!")
        print(f"轮廓内像素数: {np.sum(contour_mask > 0)}")
        print("=" * 60)
        '''
        pixel_count = int(np.sum(contour_mask > 0))
        #return contour_mask, pixel_count 这里return了两个
        return pixel_count

# run
if __name__ == "__main__":

    image_path = "../../../data/test_data/001.png"

    try:
        extractor = GreenContourExtractor(image_path)
        #mask = extractor.extract(debug=False, save_path="green_contour_mask.png") #增加像素数的输出后这行被注释掉了，不用管。
        #mask, pixel_count = extractor.extract(debug=False, save_path="green_contour_mask.png") #这个对应的是return两个的版本
        pixel_count = extractor.extract(debug=False)
        print(f"像素总数: {pixel_count}")

    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()