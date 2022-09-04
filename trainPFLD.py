import os
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset
import torchvision as tv
import cv2

#分别获取图像数据路径和landmark
def generateData(fileList):
    with open(fileList, 'r') as f:
        lines = f.readlines()
    filenames, landmarks,attributes = [], [], []
    for line in lines:
        line = line.strip().split()
        path = line[0]  #图像路径名
        landmark = np.asarray(line[1:137], dtype=np.float32) #68个坐标点的x,y
        attribute = np.asarray(line[137:], dtype=np.int32)  #属性
        filenames.append(path)
        landmarks.append(landmark)
        attributes.append(attribute)

    filenames = np.asarray(filenames, dtype=np.str)
    landmarks = np.asarray(landmarks, dtype=np.float32)
    attributes = np.asarray(attributes, dtype=np.int32)
    return (filenames, landmarks, attributes)

class DataSet(Dataset):
    def __init__(self, fileList, imageSize, transforms=None,
                 loader=tv.datasets.folder.default_loader, is_train=True):
        self.fileList, self.landmarks, self.attributes = generateData(fileList)
        self.imageSize = imageSize
        self.transforms = transforms
        self.loader = loader
        self.isTrain = is_train

    def __len__(self):
        return len(self.fileList)

    def __getitem__(self, index):
        imgPath = self.fileList[index]
        landmarks = self.landmarks[index]
        attributes = self.attributes[index]
        image = self.loader(imgPath)
        if self.transforms is not None:
            image = self.transforms(image)

        return image, landmarks, attributes


