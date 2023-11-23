## 如何修改animatediff
1. 将根目录下的motion_module.py拷贝进scripts文件夹。
2. 将scripts下的文件，其中每个py文件里面的绝对导入改为相对路径，例如：
```
scripts.animatediff_logger => .animatediff_logger
```
即可。

## hack的代码
hack代码不多，主要在animatediff_utils.py中，

1. 重写AnimateDiffControl，为了可以获取批处理的图片，从视频中。
2. 重写AnimateDiffMM，为了加载模型。
3. 重写AnimateDiffProcess与AnimateDiffI2VLatent，为了image2video的合理的特征保留。
4. 重写AnimateDiffScript，为了easyphoto更简单的调用。