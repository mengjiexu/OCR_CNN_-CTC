# OCR_CNN_-CTC
使用CNN全卷积神经网络进行整行汉字识别，正确率90-94%，这项目是我毕设的一个子项目，主要用于识别建筑图纸中的文字。
## train
requirements:
* python > 3.x
* tensorflow  1.13.1
* opencv 
* matplotlib


## 效果
训练数据为200万的类似下面的图片：
![](https://github.com/mengjiexu/OCR_CNN_-CTC/raw/master/images_result/train.png)
测试图片：
![](https://github.com/mengjiexu/OCR_CNN_-CTC/raw/master/images_result/test.jpg)
测试结果：
![](https://github.com/mengjiexu/OCR_CNN_-CTC/raw/master/images_result/test2.png)


## 训练数据
> 训练数据生成时使用的是“最新字典3_只含常用字.csv”,这里面包含的是1806个常用的汉字与建筑专业常用汉字;
> 使用的字体是收集的60多个字体文件，在项目中给出了。

## 代码
训练和测试的代码都在fcnn_ocr_v2.ipynb中
使用的是jupyter notebook进行编写的，因为方便测试和输出记录
这段时间比较忙，过一段时间进行代码重构，调用会方便很多

## 模型
训练模型使用的是GTX1080Ti，模型构建是使用的是CNN+CTC的架构，之前试过CRNN+CTC的架构，速度慢到让我抓狂，训练至少2天才能收敛，果断放弃。
在测试过多种模型结构后选择了当前的模型结构，速度方面相比CRNN快很多很多，效果对于打印体来说比较好，但是使用CNN+CTC的缺点也是显而易见的，由于视野有限，太过宽的字符是识别不出来的，好在汉字是方块字。

本人email：1596446455@qq.com ,欢迎交流



