# 将数据集按标签分成为两个文件
import shutil

dataPath = "./images/"
destPath = "./Photos/"

if __name__ == '__main__':
    with open(dataPath + 'male_names.txt') as male:
        male_names = male.readlines()
        for i in range(len(male_names) - 1):
            male_names[i] = male_names[i].strip()
            picPath = dataPath + "image/" + male_names[i]
            shutil.copyfile(picPath, destPath + "male/" + male_names[i])
    
    with open(dataPath + 'female_names.txt') as female:
        female_names = female.readlines()
        for i in range(len(female_names) - 1):
            female_names[i] = female_names[i].strip()
            picPath = dataPath + "image/" + female_names[i]
            shutil.copyfile(picPath, destPath + "female/" + female_names[i])