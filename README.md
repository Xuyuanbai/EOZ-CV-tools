# 项目简介
基于 OpenCV 开发的SMILE术后有效光学区大小识别与分析工具，计算有效光学区的基于Pentacam切向曲率差异为0的区域围成的面积。
- 自动识别地形图中的绿色区域轮廓
- 支持PNG格式输入
- 结果输出至output目录

！此项目目前并不完善，暂未提供自动安装所需软件包和手工取色的逻辑，如果运行报错可尝试修改绿色和蓝色的色值。计划在后期添加手动取色功能。

# 快速使用
> 项目基于Python3.12，克隆项目后，需先根据requirement.txt配置所需的软件包环境。

```bash
pip install -r requirements.txt
```

1. 需要提前获取术前术后切向曲率图的2 Exams Compare的截图，将所有截图放在data/RAW目录下。
2. 以下内容推荐使用面积法，*建议先了解如何修改调用图片的路径*。
  1. 单个图片计算：
     1. 面积法：运行scr/Pipeline.py进行单个EOZ直径计算；
     2. 传统6-line法：运行scr/Pipeline-6Line.py进行单个EOZ直径计算。
  2. 批量图片计算：
     1. 面积法：运行scr/main_batch.py进行批量计算，并导出为csv文件；
     2. 传统6-line法：运行scr/6-Line_batch.py进行批量计算，并导出为csv文件。
3. 导出的csv默认位于output目录中。

# TODO LIST
- [ ] 添加自动配置环境逻辑
- [ ] 优化输入与输出步骤
- [ ] readme文档添加科学性与准确性验证结果
- [ ] ...
