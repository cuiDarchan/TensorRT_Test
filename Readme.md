# Pytorch导出onnx模型，C++转化为TensorRT并实现推理

## 一. 使用指南：
**完整内容详解参考**：

1. Pytorch导出onnx模型
```
python export_onnx.py
```

2. onnxruntime推理测试
```
python onnxruntime_test.py
```

3. onnx模型转换为tensorrt模型
```
mkdir build && cd build
cmake ..
make -j8
./build_model
```

4. TensorRT模型推理测试
```
./model_infer
```

##　二. 参考： 
[1. Pytorch导出onnx模型，C++转化为TensorRT并实现推理过程](https://blog.csdn.net/weixin_44966641/article/details/125472418?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_title~default-0-125472418-blog-124557053.pc_relevant_default&spm=1001.2101.3001.4242.1&utm_relevant_index=3#t2)
[2. onnxruntime安装与使用（附实践中发现的一些问题）](https://blog.csdn.net/qq_43673118/article/details/123281548)
