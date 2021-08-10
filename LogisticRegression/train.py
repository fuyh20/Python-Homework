#导入自定义的数据加载包
from DataLoader import DataLoader 
#导入依赖的系统包
from scipy.special import expit
import numpy as np

#定义逻辑回归类
class LogisticRegression:
    
    w = None

    #训练函数
    #data_x: 训练数据集合
    #data_y: 训练数据标签集合，这里采用onehot形式
    #learning_rate: 学习速率
    #iteration_count: 迭代次数
    def train(self, data_x, data_y, learning_rate, iteration_count):
        #data_x的维度为(数据个数*数据向量宽度)
        #data_y的维度为(数据个数)
        data_x_width = len(data_x[0])   #数据向量宽度
        dataCount = len(data_y)         #数据个数

        self.w = np.zeros((data_x_width + 1, 1))          #用全零矩阵初始化权重

        data_x = np.insert(data_x, data_x_width, values=1, axis=1)  #数据中增加偏置项
        j_w = np.zeros((iteration_count, 1))               #用于存储损失结果

        real_y = np.array(data_y).reshape((dataCount, 1))            #获取二分类标签，直接获取onehot形式的标签的列向量
        for j in range(iteration_count):
            w_temp = self.w[:,0].reshape((data_x_width + 1, 1))      #获取对当前二分类项的权重，data_x_width + 1表示多一位的偏置项
            h_w = expit(np.dot(data_x,w_temp))                  #计算分类概率
            j_w[j, 0] = (np.dot(np.log(h_w).T, real_y) + np.dot((1 - real_y).T, np.log(1 - h_w))) / (-dataCount) #计算损失
            if(j_w[j, 0]> j_w[j-1, 0] and j > 1 ):
                learning_rate = learning_rate * 0.8
            print(f"learning_rate: {learning_rate}     cost: {j_w[j, 0]}")
            w_temp = w_temp + learning_rate * np.dot(data_x.T, (real_y - h_w)) #梯度下降，自动调节权重
            self.w[:, 0] = w_temp.reshape((data_x_width+1, ))         #更新对当前二分类项的权重
        

    #预测函数
    #data_x: 训练数据
    #data_y: 训练数据标签，这里采用onehot形式
    def predict(self, data_x, data_y):
        data_x_width = len(data_x[0])   #数据向量宽度
        dataCount = len(data_y)         #数据个数
        
        errorCount = 0 #预测错误的数量

        data_x = np.insert(data_x, data_x_width, values=1, axis=1)  #数据中增加偏置项
        h_w = expit(np.dot(data_x, self.w))                         #计算分类概率
        h_w = np.around(h_w)                          
        for i in range(dataCount):                                  #统计预测错误的数量
            if data_y[i] != h_w[i]:
                errorCount += 1

        error_rate = float(errorCount) / dataCount #计算错误率

        return error_rate #返回错误率


if __name__ == '__main__':

    #加载Mnist数据
    print('Loading data...')
    dataLoader = DataLoader()
    dataLoader.loadData()

    #获取训练、验证和测试数据
    train_images, train_labels = dataLoader.getTrainData()
    validation_images, validation_labels = dataLoader.getValidationData()
    test_images, test_labels = dataLoader.getTestData()


    image_width = len(train_images[0])

    learning_rate = 0.000005    #LogisticRegression的学习速率参数


    print('Start training...')
    lr = LogisticRegression()
    lr.train(train_images, train_labels, learning_rate, 5000)     #使用训练集训练
    error_rate = lr.predict(validation_images, validation_labels) #使用验证集预测
    print('Validation error rate: ', error_rate)
    print('Train done')

    error_rate = lr.predict(test_images, test_labels)             #使用测试集预测最终结果
    print('Test error rate: ', error_rate)

    np.savetxt('LR-parameters.txt', lr.w, delimiter=',')