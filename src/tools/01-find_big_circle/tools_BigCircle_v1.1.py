#pre: 在V1的基础上，基于V1生成的刺突掩码通过举例算法寻找圆心和半径
#实现： for pixel in mask, distance <- distance.mini(pixel to boarder), r <- max(distance) ！！！重要逻辑！！！
#post: 所得轮廓比肉眼分辨的小一圈，新问题：如何寻找确切地mask边界

#way1: 修改开闭运算的阈值，把第三步二值化35->20，效果不显著。
#way2:为mask人为设定“膨胀”，mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)，可以修改为6或7。当前抽样验证：7满足要求。
#post-way2：基本满足需求。但是存在一个问题：为什么选择6？而不选择更大或更小的值？除了抽样5张图片之外，能否推广到所有图像？

import cv2
import numpy as np
import os

def detect_topography_circle(image_path):
    # 1. input
    if not os.path.exists(image_path):
        print(f"错误：找不到文件 {image_path}")
        return

    img = cv2.imread(image_path)
    original = img.copy()

    # 2. blur
    blurred = cv2.medianBlur(img, 7)

    # 3. lab空间提取
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2Lab)
    a_chan = lab[:, :, 1].astype(float)
    b_chan = lab[:, :, 2].astype(float)
    dist = np.sqrt((a_chan - 128) ** 2 + (b_chan - 128) ** 2)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 二值化
    _, thresh = cv2.threshold(dist, 35, 255, cv2.THRESH_BINARY)

    # 4. fill the circle
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)

    # 5. core：diatamce 算法 - mini and max
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)
    center = max_loc  # 圆心坐标 (x, y)
    radius = max_val  # 半径

    if radius <= 0:
        print("未检测到有效圆形区域")
        return

    # 6. output
    debug_img = original.copy()
    cv2.circle(debug_img, center, int(radius), (0, 0, 255), 3)
    cv2.circle(debug_img, center, 3, (0, 255, 0), -1)

    print("-" * 30)
    print(f"检测成功！")
    print(f"圆心坐标: {center}")
    print(f"所得半径: {radius:.2f} 像素")
    print("-" * 30)

    cv2.imshow("1. Color Mask (Input)", mask)
    cv2.imshow("2. Debug Result (Red Circle)", debug_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# run
path = '../../../tests/001.png'  # 请确保文件名正确
detect_topography_circle(path)