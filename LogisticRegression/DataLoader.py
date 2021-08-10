from PIL import Image
import numpy as np

class DataLoader:
    images = []
    labels = []

    dataPath = './Dataset/image/'

    train_images = []
    train_labels = []
    
    validation_images = []
    validation_labels = []

    test_images = []
    test_labels = []

    dataCount = 13234

    #加载图片数据
    def loadData(self):
        with open('./Dataset/male_names.txt') as male:
            male_names = male.readlines()
            for i in range(len(male_names) - 1):
                male_names[i] = male_names[i].strip()
                picPath = self.dataPath + male_names[i]
                image = self.loadPicArray(picPath)
                label = 0
                self.images.append(image)
                self.labels.append(label)

        with open('./Dataset/female_names.txt') as female:
            female_names = female.readlines()
            for i in range(len(female_names) - 1):
                female_names[i] = female_names[i].strip()
                picPath = self.dataPath + female_names[i]
                image = self.loadPicArray(picPath)
                label = 1
                self.images.append(image)
                self.labels.append(label)
        
        
        #打乱数据，使用相同的次序打乱images、labels保证数据仍然对应
        state = np.random.get_state()
        np.random.shuffle(self.images)
        np.random.set_state(state)
        np.random.shuffle(self.labels)

        #按比例切割数据，分为训练集、验证集和测试集
        trainIndex = int(self.dataCount * 0.4)
        validationIndex = int(self.dataCount * 0.5)
        self.train_images = self.images[0 : trainIndex]
        self.train_labels = self.labels[0 : trainIndex]
        self.validation_images = self.images[trainIndex : validationIndex]
        self.validation_labels = self.labels[trainIndex : validationIndex]
        self.test_images = self.images[validationIndex : ]
        self.test_labels = self.labels[validationIndex : ]

      #读取图片数据，得到图片对应的像素值的数组，均一化到0-1之前
    def loadPicArray(self, picFilePath):
        picData = Image.open(picFilePath)
        ImgGray = picData.convert('L')
        picArray = np.array(ImgGray).flatten() / 255.0
        return picArray

    def getTrainData(self):
        return self.train_images, self.train_labels

    def getValidationData(self):
        return self.validation_images, self.validation_labels

    def getTestData(self):
        return self.test_images, self.test_labels   