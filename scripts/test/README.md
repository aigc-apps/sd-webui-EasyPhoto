# Test & Profile-23/10/09
## 环境准备
- 请搭建一个待测试的SDWebUI环境
- 保证可以访问到上述环境的情况下，准备python环境，满足依赖 base64/json/numpy/cv2/

## 训练/推理测试代码
- **post_train.py** 支持公网URL图片/本地图片读取
- **post_infer.py** 支持公网URL图片/本地图片读取
代码提供了默认URL，可修改，本地图片通过命令行参数输入。


## 测试平均耗时 32G-V100
### 训练
**初始模型下载完成后**，avg_time ~= 7mins， **max_train_steps**=100，参考下图


![train](https://pai-vision-data-inner.oss-accelerate.aliyuncs.com/data/easyphoto/reademe/train.png)

### 推理
**初始模型下载完成后** avg_infer_time(10次) ~= 20s


![infer](https://pai-vision-data-inner.oss-accelerate.aliyuncs.com/data/easyphoto/reademe/infer.png)