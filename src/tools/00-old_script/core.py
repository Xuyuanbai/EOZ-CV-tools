import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

#
class TerrainMapAnalyzer:

    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"No Image: {image_path}")
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.outer_diameter_mm = 9.0
        self.pixel_to_mm_ratio = None

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
        """检测大圆边界（彩色区域预筛选 + 轮廓检测）"""
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
            diameter_pixels = 2 * r
            self.pixel_to_mm_ratio = self.outer_diameter_mm / diameter_pixels

            if debug:
                print(f"大圆检测 - 圆心: ({x}, {y}), 半径: {r}像素")
                print(f"直径: {diameter_pixels}像素 = {self.outer_diameter_mm}mm")
                print(f"转换比例: 1像素 = {self.pixel_to_mm_ratio:.6f}mm")

            return x, y, r
        else:
            raise ValueError("未能检测到大圆边界")

    def is_green_pixel(self, pixel_hsv):
        """判断像素是否为绿色（精确匹配）"""
        for green_color in self.green_colors:
            if np.array_equal(pixel_hsv, green_color):
                return True
        return False

    def is_blue_pixel(self, pixel_hsv):
        """判断像素是否为蓝色（精确匹配）"""
        for blue_color in self.blue_colors:
            if np.array_equal(pixel_hsv, blue_color):
                return True
        return False

    def detect_green_band_boundary(self, center_x, center_y, outer_radius, debug=False):
        """
        检测绿色色带的外边界（精确颜色匹配：绿色→蓝色过渡点）
        """
        height, width = self.hsv.shape[:2]
        inner_mask = np.zeros((height, width), dtype=np.uint8)

        # 设置最大扫描半径
        max_scan_radius = int(outer_radius * 0.8)

        if debug:
            print(f"最大扫描半径: {max_scan_radius}像素")

        # 径向扫描
        num_angles = 360
        angles = np.linspace(0, 2 * np.pi, num_angles)
        boundary_radii = []

        # 诊断计数器
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

                # 诊断计数
                if is_green:
                    green_found_count += 1
                if is_blue:
                    blue_found_count += 1

                # 记录绿色出现
                if is_green:
                    green_started = True

                # 检测绿色→蓝色过渡（绿色消失且蓝色出现）
                if green_started and is_blue:
                    green_end_radius = r
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
            raise ValueError(f"有效边界点太少({len(valid_radii)})，无法分析")

        # 统计离群值剔除（IQR方法）
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

        # 标记离群值为-1
        valid_indices = np.where(boundary_radii != -1)[0]
        for i, idx in enumerate(valid_indices):
            if outlier_mask[i]:
                boundary_radii[idx] = -1

        # 插值处理
        missing_count = np.sum(boundary_radii == -1)

        if missing_count > 0:
            normal_indices = np.where(boundary_radii != -1)[0]
            normal_radii = boundary_radii[normal_indices]

            if len(normal_indices) < 3:
                raise ValueError("正常值太少，无法插值")

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

        # 生成边界点
        boundary_points = []
        for i, angle in enumerate(angles):
            r = boundary_radii[i]
            x = int(center_x + r * np.cos(angle))
            y = int(center_y + r * np.sin(angle))
            x = np.clip(x, 0, width - 1)
            y = np.clip(y, 0, height - 1)
            boundary_points.append((x, y))

        # 填充内圈
        contour_points = np.array(boundary_points, dtype=np.int32)
        cv2.fillPoly(inner_mask, [contour_points], 255)

        if debug:
            print(f"边界点数: {len(boundary_points)}")
            print(f"内圈像素数: {np.sum(inner_mask > 0)}")
            if missing_count > 0:
                print(f"已插值修复 {missing_count} 个角度（含{outlier_count}个离群值）")

        return inner_mask

    def calculate_area_and_diameter(self, inner_mask):
        """计算面积和等效圆直径"""
        pixel_count = np.sum(inner_mask > 0)
        pixel_area_mm2 = self.pixel_to_mm_ratio ** 2
        area_mm2 = pixel_count * pixel_area_mm2
        equivalent_diameter_mm = 2 * np.sqrt(area_mm2 / np.pi)

        return area_mm2, equivalent_diameter_mm

    def visualize_results(self, center_x, center_y, outer_radius,
                          inner_mask, save_path=None):
        """可视化分析结果"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 原图
        axes[0].imshow(self.image_rgb)
        axes[0].set_title('原始图像', fontsize=14)
        axes[0].axis('off')

        # 绿色内圈掩码
        axes[1].imshow(inner_mask, cmap='gray')
        axes[1].set_title('绿色内圈区域（白色部分）', fontsize=14)
        axes[1].axis('off')

        # 标注结果
        result_img = self.image_rgb.copy()

        # 画大圆参考
        cv2.circle(result_img, (center_x, center_y), outer_radius,
                   (255, 0, 255), 2)

        # 叠加绿色内圈
        overlay = result_img.copy()
        green_overlay = np.zeros_like(result_img)
        green_overlay[inner_mask > 0] = [0, 255, 0]
        overlay = cv2.addWeighted(overlay, 1.0, green_overlay, 0.4, 0)

        # 绘制边界
        contours, _ = cv2.findContours(inner_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        axes[2].imshow(overlay)
        axes[2].set_title('分析结果（绿色区域为计算面积）', fontsize=14)
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"结果已保存到: {save_path}")

        plt.show()

    def analyze(self, debug=True, visualize=True, save_path=None):
        """执行完整分析流程"""
        print("=" * 60)
        print("开始分析地形图（v6版本 - 实际采样精确颜色）...")
        print("=" * 60)

        # 1. 检测大圆边界
        print("\n[步骤1] 检测大圆边界（彩色预筛选 + 轮廓检测）...")
        center_x, center_y, outer_radius = self.detect_outer_circle(debug=debug)

        # 2. 检测绿色色带边界
        print("\n[步骤2] 检测绿色色带边界（精确颜色匹配）...")
        print("  - 绿色: HSV(68,231,210) 和 HSV(48,255,230)")
        print("  - 蓝色: 4个精确采样值")
        print("  - 边界检测: 绿色消失→蓝色出现")
        inner_mask = self.detect_green_band_boundary(
            center_x, center_y, outer_radius, debug=debug
        )

        # 3. 计算面积
        print("\n[步骤3] 计算实际面积和等效圆直径...")
        area_mm2, equiv_diameter_mm = self.calculate_area_and_diameter(inner_mask)
        equiv_radius_mm = equiv_diameter_mm / 2

        # 输出结果
        print("\n" + "=" * 60)
        print("分析结果:")
        print("=" * 60)
        print(f"绿色内圈包围的实际面积(S): {area_mm2:.4f} mm²")
        print(f"等效圆直径: {equiv_diameter_mm:.4f} mm")
        print(f"等效圆半径: {equiv_radius_mm:.4f} mm")
        print(f"像素总数: {np.sum(inner_mask > 0)}")
        print("=" * 60)

        # 4. 可视化
        if visualize:
            print("\n[步骤4] 生成可视化结果...")
            self.visualize_results(center_x, center_y, outer_radius,
                                   inner_mask, save_path)

        return {
            'area_mm2': area_mm2,
            'equivalent_diameter_mm': equiv_diameter_mm,
            'equivalent_radius_mm': equiv_radius_mm,
            'pixel_count': int(np.sum(inner_mask > 0)),
            'center': (center_x, center_y),
            'pixel_to_mm_ratio': self.pixel_to_mm_ratio,
            'inner_mask': inner_mask
        }



if __name__ == "__main__":
    image_path = "../../../data/RAW/641396800.png"

    try:
        analyzer = TerrainMapAnalyzer(image_path)
        results = analyzer.analyze(debug=True, visualize=True,
                                   save_path="analysis_result_v6.png")

        print("\n分析完成！")
        print(f"面积: {results['area_mm2']:.4f} mm²")

    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()