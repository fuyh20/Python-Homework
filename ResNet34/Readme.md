# README

* data_classification.py：将数据集按标签分为两个文件夹
* main.py：加载训练好的模型，输入./images/中的文件名，来预测性别
* model.py：定义ResNet34
* my_dataset.py：自定义了数据集类
* train.py：训练模型
* utils.py：定义了方法划分数据集，可视化

该目录存放数据集，需要有两个文件夹（首先运行data_classification.py）：

**./images/**中有所有的文件和female.txt、male.txt           

**./photos/**中有两个文件夹**female、male**

