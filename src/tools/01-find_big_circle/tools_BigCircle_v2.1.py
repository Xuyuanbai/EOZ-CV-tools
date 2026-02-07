#pre：拟合圆形
#post：在gemini帮助下，实现了圆的拟合（最长轮廓+最小二乘拟合），至于实现这一过程的什么浮点运算，我自己也不太懂。T_T...
#final：**弃用这一版**！！！

import cv2
import numpy as np


def fit_circle_from_gray_edge(image_path):
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

# 下面是gemini给出的部分，因为我个人已经凌乱了。。。

    # 将坐标转为浮点数格式以供拟合？？？
    #pts = points.reshape(-1, 2).astype(np.float32) #gemini提供的这一行没有用到，我训问他之后，意思是我下面的方案没用到这个点集法。我也不懂，服了这里。

    # 使用 OpenCV 提供的圆形拟合 (HoughCircles 或 MinEnclosingCircle)
    # 这里我们采用一种更稳健的方法：找最大轮廓并拟合
    contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最长的那个轮廓（通常就是我们要的圆圈）
    #c = max(contours, key=cv2.arcLength) #这一行会报错，询问gemini后得知是因为没有确定轮廓是否闭合。
    # True 表示我们认为我们要找的那个最长轮廓是闭合的
    c = max(contours, key=lambda x: cv2.arcLength(x, True))

    # 拟合最小二乘圆
    (x, y), radius = cv2.minEnclosingCircle(c)
    center = (int(x), int(y))
    radius = int(radius)

    # 绘制结果
    result_img = img.copy()
    cv2.circle(result_img, center, radius, (0, 0, 255), 2)  # 红色圆周
    cv2.circle(result_img, center, 3, (255, 0, 0), -1)  # 蓝色圆心

    print("-" * 30)
    print(f"【拟合成功】")
    print(f"圆心: {center}")
    print(f"精准半径: {radius:.2f}")
    print("-" * 30)

    cv2.imshow("Fitting Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# run
fit_circle_from_gray_edge('../../../data/test_data/001.png')