#pre：尝试通过色彩运算消除坐标文字的影响。
#post：生成的包络仍然受文字影响。且无法得出圆形，得到的是圆形+半岛样刺突。
#then：不进行凸包，而是：寻找mask的最大内切圆，或直接寻找圆形的半径（v2）

import cv2
import numpy as np
import os


def detect_topography_contour(image_path):
    # 1. input
    if not os.path.exists(image_path):
        print(f"错误：找不到文件 {image_path}")
        return

    img = cv2.imread(image_path)
    original = img.copy()

    # 2. blur消除噪点？
    blurred = cv2.medianBlur(img, 7)

    # 3. lab空间提取色彩显著，排除128
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2Lab)
    a_chan = lab[:, :, 1].astype(float)
    b_chan = lab[:, :, 2].astype(float)

    # 计算偏离中性色的欧氏距离，因为环内色彩鲜艳所有distance大
    dist = np.sqrt((a_chan - 128) ** 2 + (b_chan - 128) ** 2)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 4. 二提取将圆环，35为阈值
    _, thresh = cv2.threshold(dist, 35, 255, cv2.THRESH_BINARY)

    # 5. 闭运算，填充和补全
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 6. 只保留面积最大的那一块 ###这里document没完全看懂，但确实需要这样做。
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    if num_labels <= 1:
        print("未检测到circle")
        return

    # 最大面积
    max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # make mask
    final_mask = np.zeros_like(mask)
    final_mask[labels == max_label] = 255

    # 7. 凸包（橡皮筋包络）算法消除缺失
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(c)  # 获取凸包

        # 8. output
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
        cv2.drawContours(img, [hull], -1, (0, 0, 255), 3)

        cv2.imshow("Original", original)
        cv2.imshow("Color Mask", final_mask)
        cv2.imshow("Detected Contour (Red is Hull)", img)

        print(f"检测成功。圆心坐标: {centroids[max_label]}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# run
path = '../../../data/test_data/001.png'
detect_topography_contour(path)