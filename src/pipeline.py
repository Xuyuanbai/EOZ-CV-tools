"""
通过简历函数简单实现对单个地形图文件的调用,
"""

import numpy as np
from BigCircle import detect_topography_circle
from GreenArea import GreenContourExtractor

def main(pathway):
    #基于Big_Circle脚本计算每个像素的实际长度
    length_pixel_circle = detect_topography_circle(pathway)
    length_per_pixel = 4.5 / length_pixel_circle
    s_per_pixel = length_per_pixel **2

    #基于GreenArea脚本计算绿色区域像素数目
    green = GreenContourExtractor(pathway)
    green_pixel_num = green.extract(debug=False)

    #数学计算
    s = green_pixel_num * s_per_pixel
    r = np.sqrt(s / np.pi)
    eoz = 2 * r
    return eoz

if __name__ == '__main__':
    d = main("../tests/确定圆5.5mm.png")
    print(d)