## miopen_conv

### 简介：

​		miopen_conv——A convolutional function is implemented by calling the API to help understand how miopen works. demo使用miopen的hip backend作为实现，通过最简单的方式实现了Immediate与Find-Db两种模式，结合最普通的direct计算方式，完成了miopen结果的验证。

### 使用：

​		1、要求:设备先安装完成ROCM、Hip、以及Miopen这些必须得依赖项。安装参考https://github.com/ROCmSoftwarePlatform/MIOpen
​		2、编译时，请修改Makefile中的--amdgpu项，选择正确的设备架构版本；

### 结果：



**默认使用immediate moel，main.cpp中实现了find-Db的功能，如需要请修改注释并重新编译完成该功能**

参考资料：
（1）https://github.com/ROCmSoftwarePlatform/MIOpen.git -b Tags rocm-5.2.0

（2）https://github.com/ROCmSoftwarePlatform/MIOpenExamples

（3）https://github.com/ROCmSoftwarePlatform/miopen_cudnn_ops


​		