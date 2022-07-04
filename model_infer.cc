// tensorrt相关
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>  //新添加，nvinfer1::AsciiChar

// onnx解析器相关
#include <NvOnnxParser.h>  // 与原文不同，onnx-tensorrt build后，sudo make install

// cuda_runtime相关
#include <cuda_runtime.h>
#include <cuda.h>

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

#ifndef checkRuntime
#define checkRuntime(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
            assert(0);\
        }\
    }
#endif  // checkRuntime

// 定义make_nvshared智能指针
template <class T>
std::shared_ptr<T> make_nvshared(T *ptr)
{
    return std::shared_ptr<T>(ptr, [](T *p)
                              { p->destroy(); });
}


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

std::vector<unsigned char> load_file(const std::string& file) {
	std::ifstream in(file, std::ios::in | std::ios::binary);
	if (!in.is_open()) return {};

	in.seekg(0, std::ios::end);
	size_t length = in.tellg();

	std::vector<uint8_t> data;
	if (length > 0) {
		in.seekg(0, std::ios::beg);
		data.resize(length);
		in.read((char*)&data[0], length);
	}
	in.close();
	return data;
}

void inference(const std::string& image_path) {
	TRTLogger logger;
  // 加载模型
	auto engine_data = load_file("../resnet50.engine");
  // 执行推理前，需要创建一个推理的runtime接口实例。与builer一样，runtime需要logger
	auto runtime = make_nvshared(nvinfer1::createInferRuntime(logger));
	auto engine = make_nvshared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
	if (engine == nullptr) {
		printf("Deserialize cuda engine failed.\n");
		runtime->destroy();
		return;
	}

	if (engine->getNbBindings() != 2) {
		printf("Must be single input, single Output, got %d output.\n", engine->getNbBindings() - 1);
		return;
	}

  // 创建CUDA流，以确定这个batch的推理是独立的
	cudaStream_t stream = nullptr;
	checkRuntime(cudaStreamCreate(&stream));
	auto execution_context = make_nvshared(engine->createExecutionContext());

	int input_batch = 1;
	int input_channel = 3;
	int input_height = 224;
	int input_width = 224;

  // 准备好input_data_host和input_data_device，分别表示内存中的数据指针和显存中的数据指针
  // 一会儿将预处理过的图像数据搬运到GPU
	int input_numel = input_batch * input_channel * input_height * input_width;
	float* input_data_host = nullptr;
	float* input_data_device = nullptr;

	checkRuntime(cudaMallocHost(&input_data_host, input_numel * sizeof(float)));
	checkRuntime(cudaMalloc(&input_data_device, input_numel * sizeof(float)));

  // 图片读取与预处理，与之前python中的预处理方式一致：
  // BGR->RGB、归一化/除均值减标准差
  float mean[] = {0.406, 0.456, 0.485};
	float std[] = {0.225, 0.224, 0.229};
  
	auto image = cv::imread(image_path);
	cv::resize(image, image, cv::Size(input_width, input_height));

	int image_area = image.cols * image.rows;
	unsigned char* pimage = image.data;
	float* phost_b = input_data_host + image_area * 0;
	float* phost_g = input_data_host + image_area * 1;
	float* phost_r = input_data_host + image_area * 2;
	for (int i=0; i<image_area; ++i, pimage += 3) {
		 *phost_r++ = (pimage[0] / 255.0f - mean[0]) / std[0];
		 *phost_g++ = (pimage[1] / 255.0f - mean[1]) / std[1];
		 *phost_b++ = (pimage[2] / 255.0f - mean[2]) / std[2];
	 }

  // 进行推理
	checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel *sizeof(float), cudaMemcpyHostToDevice, stream));

	const int num_classes = 1000;
	float output_data_host[num_classes];
	float* output_data_device = nullptr;
	checkRuntime(cudaMalloc(&output_data_device, sizeof(output_data_host)));

	auto input_dims = engine->getBindingDimensions(0);
	input_dims.d[0] = input_batch;

	execution_context->setBindingDimensions(0, input_dims);
  // 用一个指针数组bindings指定input和output在gpu中的指针。
	float* bindings[] = {input_data_device, output_data_device};
	bool success = execution_context->enqueueV2((void**)bindings, stream, nullptr);

	checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream));
	checkRuntime(cudaStreamSynchronize(stream));

	float* prob = output_data_host;
	int predict_label = std::max_element(prob, prob + num_classes) - prob;
	float conf = prob[predict_label];
	printf("test_image: %s, max_idx: %d, probability: %f", image_path.c_str(), predict_label, conf);

  // 释放显存
	checkRuntime(cudaStreamDestroy(stream));
	checkRuntime(cudaFreeHost(input_data_host));
	checkRuntime(cudaFree(input_data_device));
	checkRuntime(cudaFree(output_data_device));
}

int main(){
    std::string image_path = "../test.jpg";
    inference(image_path);
    return 0;
}