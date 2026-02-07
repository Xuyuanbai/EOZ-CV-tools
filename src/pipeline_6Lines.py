from BigCircle import detect_topography_circle
from Green6Lines import GreenDiameterMeasurer

def main(pathway):
    #基于Big_Circle脚本计算每个像素的实际长度
    length_pixel_circle = detect_topography_circle(pathway)
    length_per_pixel = 4.5 / length_pixel_circle

    #计算绿色经线平均像素长度
    measurer = GreenDiameterMeasurer(pathway)
    avg_diameter_px = measurer.measure_diameters()

    #数学计算
    eoz = avg_diameter_px *length_per_pixel
    return eoz, length_pixel_circle

if __name__ == '__main__':
    d, a = main("../tests/006-.png")
    print(d)
    print(a)