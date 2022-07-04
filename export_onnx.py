# export_onnx.py
import torch
import torchvision.models as models
import cv2
import numpy as np

class ResNet50_wSoftmax(torch.nn.Module):
    # 将softmax后处理合并到模型中，一起导出为onnx
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        y = self.base_model(x)
        prob = self.softmax(y)
        return prob

def preprocessing(img):
    # 预处理：BGR->RGB、归一化/除均值减标准差
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    img = img[:, :, ::-1]
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1).astype(np.float32)
    tensor_img = torch.from_numpy(img)[None] # 此处增加一维
    return tensor_img

if __name__ == '__main__':
    # model = models.resnet50(pretrained=True)
    image_path = 'test.jpg'
    img = cv2.imread(image_path)
    tensor_img = preprocessing(img)
    model = ResNet50_wSoftmax()   # 将后处理添加到模型中
    model.eval()
    pred = model(tensor_img)[0]
    max_idx = torch.argmax(pred)
    print(f"test_image: {image_path}, max_idx: {max_idx}, max_logit: {pred[max_idx].item()}")

    dummpy_input = torch.zeros(1, 3, 224, 224)  # onnx的导出需要指定一个输入，这里直接用上面的tenosr_img也可
    torch.onnx.export(
            model, dummpy_input, 'resnet50_wSoftmax.onnx',
            input_names=['image'],
            output_names=['predict'],
            opset_version=11,
            dynamic_axes={'image': {0: 'batch'}, 'predict': {0: 'batch'}}		# 注意这里指定batchsize是动态可变的
    )
