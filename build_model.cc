// tensorrt相关
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>  //新添加，nvinfer1::AsciiChar

// onnx解析器相关
#include <NvOnnxParser.h>  // 与原文不同，onnx-tensorrt build后，sudo make install

// cuda_runtime相关
#include <cuda_runtime.h>

// 常用头文件
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <dirent.h>

// opencv
#include <opencv2/opencv.hpp>

inline const char* severity_string(nvinfer1::ILogger::Severity t) {
  switch (t) {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
      return "internal_error";
    case nvinfer1::ILogger::Severity::kERROR:
      return "error";
    case nvinfer1::ILogger::Severity::kWARNING:
      return "warning";
    case nvinfer1::ILogger::Severity::kINFO:
      return "info";
    case nvinfer1::ILogger::Severity::kVERBOSE:
      return "verbose";
    default:
      return "unknown";
  }
}

class TRTLogger : public nvinfer1::ILogger {
 public:
  virtual void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      if (severity == Severity::kWARNING)
        printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
      else if (severity == Severity::kERROR)
        printf("\031[33m%s: %s\033[0m\n", severity_string(severity), msg);
      else
        printf("%s: %s\n", severity_string(severity), msg);
    }
  }
};

bool build_model() {
  if (access("resnet50.engine", 0) == 0) {
    printf("resnet50.engine already exists.\n");
    return true;
  }

  TRTLogger logger;

  // 下面的builder, config, network是基本需要的组件
  // 形象的理解是你需要一个builder去build这个网络，网络自身有结构，这个结构可以有不同的配置
  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
  // 创建一个构建配置，指定TensorRT应该如何优化模型，tensorRT生成的模型只能在特定配置下运行
  nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
  // 创建网络定义，其中createNetworkV2(1)表示采用显性batch
  // size，新版tensorRT(>=7.0)时，不建议采用0非显性batch size
  nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);

  // onnx parser解析器来解析onnx模型
  auto parser = nvonnxparser::createParser(*network, logger);
  if (!parser->parseFromFile("../resnet50_wSoftmax.onnx", 1)) {
    printf("Failed to parse resnet50_wSoftmax.onnx.\n");
    return false;
  }

  // 设置工作区大小
  printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
  config->setMaxWorkspaceSize(1 << 28);

  // 需要通过profile来使得batchsize时动态可变的，这与我们之前导出onnx指定的动态batchsize是对应的
  int maxBatchSize = 10;
  auto profile = builder->createOptimizationProfile();
  auto input_tensor = network->getInput(0);
  auto input_dims = input_tensor->getDimensions();

  // 设置batchsize的最大/最小/最优值
  input_dims.d[0] = 1;
  profile->setDimensions(input_tensor->getName(),
                         nvinfer1::OptProfileSelector::kMIN, input_dims);
  profile->setDimensions(input_tensor->getName(),
                         nvinfer1::OptProfileSelector::kOPT, input_dims);

  input_dims.d[0] = maxBatchSize;
  profile->setDimensions(input_tensor->getName(),
                         nvinfer1::OptProfileSelector::kMAX, input_dims);
  config->addOptimizationProfile(profile);

  // 开始构建tensorrt模型engine
  nvinfer1::ICudaEngine* engine =
      builder->buildEngineWithConfig(*network, *config);

  if (engine == nullptr) {
    printf("Build engine failed.\n");
    return false;
  }

  // 将构建好的tensorrt模型engine反序列化（保存成文件）
  nvinfer1::IHostMemory* model_data = engine->serialize();
  FILE* f = fopen("resnet50.engine", "wb");
  fwrite(model_data->data(), 1, model_data->size(), f);
  fclose(f);

  // 逆序destory掉指针
  model_data->destroy();
  engine->destroy();
  network->destroy();
  config->destroy();
  builder->destroy();

  printf("Build Done.\n");
  return true;
}

int main() {
  if (!build_model()) {
    printf("Couldn't build engine!\n");
  }
  return 0;
}