#pre：尝试得到色彩为（128，128，128）的灰色轮廓
#post：得到了完美的外圆+比例尺图形：file output v2.0，需要在V2.1中拟合圆形

import cv2
import numpy as np


def extract_gray_edge(image_path):
    # 1. input
    img = cv2.imread(image_path)
    if img is None:
        print("错误：无法读取图片")
        return

    target_gray = np.array([128, 128, 128])

    # 是否增加“包容”性？记得取消注释符号
    tolerance = 2
    lower_gray = target_gray - tolerance
    upper_gray = target_gray + tolerance

    # 2. make mask
    #gray_mask = cv2.inRange(img, lower_gray, upper_gray)
    #不增加包容性 (经验证，对目前取样没有影响，记得取消注释符号）
    gray_mask = cv2.inRange(img, target_gray, target_gray)
    # 3. 得到点
    points = cv2.findNonZero(gray_mask)
    if points is not None:
        print(f"检测到符合条件的灰度点数量: {len(points)}")
    else:
        print("未检测到符合条件的灰度点")
        return

    # 4. putput
    cv2.imshow("Original", img)
    cv2.imshow("Detected Gray Points (128, 128, 128)", gray_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# run
extract_gray_edge('../../../data/test_data/001.png')