## 环境准备
请确保测试的环境中安装了以下扩展包
- base64
- json
- cv2
- numpy
- datetime

## 训练测试代码
详情代码请看**post_train.py**文件。其中提供两种图片导入方式，第一种是公网图片URL，第二种是本地图片目录读取
### 公网图片URL
代码中有写到，将你要训练的若干张图片链接，填入列表里面，进行post请求时，后端训练会接受到若干图片。</br>
运行代码时，直接运行代码即可
> python post_trian.py
### 本地目录图片
代码中有写到，将你要训练的若干张图片，放入本地某个目录中，**并保证全部目录下面全部是可读取的图片**，进行post请求时，后端训练会接受到若干图片。</br>
运行代码时，加个图片目录路径参数
> python post_trian.py  ./xxxxx

## 推理测试代码
详情代码请看**post_infer.py**文件，其中提供两种图片导入方式，第一种是公网图片URL，第二种是本地图片路径。
### 公网图片URL
代码中有写到，将你要推理的某张模板图片链接，填入对应变量中，进行post请求时，后端推理服务会接受到该图片。</br>
运行代码时，直接运行代码即可
> python post_infer.py

### 本地目录图片
代码中有写到，将你要推理的某张模板图片的本地路径，作为一个参数跟随代码文件的后面，进行post请求时，后端推理会接受到此图片。</br>
运行代码时，加个模板图片路径参数
> python post_infer.py  ./xxxxx.jpg

## 测试平均耗时
### 训练
在训练条件具备的情况下（比如该下载的模型都下好的情况下）测试训练平均耗时大约7分钟左右，最大训练 **max_train_steps** 参数设置成100，训练耗时见如图


![train](https://pai-vision-data-inner.oss-accelerate.aliyuncs.com/data/easyphoto/reademe/train.png)

### 推理
在不切换模型的情况下，测试推理平均耗时20s左右，推理耗时见如图


![infer](https://pai-vision-data-inner.oss-accelerate.aliyuncs.com/data/easyphoto/reademe/infer.png)