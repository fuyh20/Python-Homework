# 输入图片的文件名，对该图片进行预测

import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import resnet34

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 加载图片
    img_root_path = "./images/"
    file_name = input("请输入图片的文件名：")
    assert os.path.exists(img_root_path + file_name), "file: '{}' dose not exist.".format(img_root_path + file_name)
    img = Image.open(img_root_path + file_name)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # 展开批处理维度
    img = torch.unsqueeze(img, dim=0)

    # 读入类别的指数
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # 建立模型
    model = resnet34(num_classes=5).to(device)

    # 加载训练好的模型参数
    weights_path = "./resNet34.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # 进行计算得到结果并进行可视化
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()