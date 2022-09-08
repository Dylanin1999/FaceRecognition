import os
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset
import torchvision as tv
import cv2
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from nets import pfldNet
from euler_angles_utils import calculate_pitch_yaw_roll
import time


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

    filenames = np.asarray(filenames, dtype=np.str_)
    landmarks = np.asarray(landmarks, dtype=np.float32)
    attributes = np.asarray(attributes, dtype=np.int32)
    return (filenames, landmarks, attributes)

def get_euler_angle_weights(landmarks_batch, euler_angles_pre, device):
    TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]

    euler_angles_landmarks = []
    landmarks_batch = landmarks_batch.numpy()
    for index in TRACKED_POINTS:
        euler_angles_landmarks.append(landmarks_batch[:, 2 * index:2 * index + 2])
    euler_angles_landmarks = np.asarray(euler_angles_landmarks).transpose((1, 0, 2)).reshape((-1, 28))

    euler_angles_gt = []
    for j in range(euler_angles_landmarks.shape[0]):
        pitch, yaw, roll = calculate_pitch_yaw_roll(euler_angles_landmarks[j])
        euler_angles_gt.append((pitch, yaw, roll))
    euler_angles_gt = np.asarray(euler_angles_gt).reshape((-1, 3))

    euler_angles_gt = torch.Tensor(euler_angles_gt).to(device)
    euler_angle_weights = 1 - torch.cos(torch.abs(euler_angles_gt - euler_angles_pre))
    euler_angle_weights = torch.sum(euler_angle_weights, 1)

    return euler_angle_weights


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
        if index == 100:
            print(imgPath)
        return image, landmarks, attributes



if __name__=='__main__':
    start_time = time.time()
    train_data_transforms = tv.transforms.Compose([
        tv.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor()
    ])

    train_dataset = DataSet("./data/list.txt", 224, transforms=train_data_transforms,
                            is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("torch.cuda.is_available()", torch.cuda.is_available())
    model = pfldNet.MobileNetV2().cuda()
    auxiliary_net = pfldNet.AuxiliaryNet().cuda()


    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': auxiliary_net.parameters()}],
                                 lr=0.1, weight_decay=5e-5)  # optimizer
    lr_epoch = '20,50,100,300,400'
    lr_epoch = lr_epoch.strip().split(',')
    lr_epoch = list(map(int, lr_epoch))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_epoch, gamma=0.1)

    for epoch in range(1000):
        # model.train()
        # auxiliary_net.train()

        for i_batch, (images_batch, landmarks_batch, attributes_batch) in enumerate(train_loader):
            images_batch = images_batch.to(device)
            landmarks_batch = landmarks_batch
            pre_landmarks, auxiliary_features = model(images_batch)
            euler_angles_pre = auxiliary_net(auxiliary_features.to(device))
            euler_angle_weights = get_euler_angle_weights(landmarks_batch, euler_angles_pre, device)
            loss = pfldNet.wing_loss(landmarks_batch.to(device), pre_landmarks, euler_angle_weights)
            loss = torch.nn.L1Loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((i_batch + 1) % 100) == 0 or (i_batch + 1) == len(train_loader):
                Epoch = 'Epoch:[{:<4}][{:<4}/{:<4}]'.format(epoch, i_batch + 1, len(train_loader))
                Loss = 'Loss: {:2.3f}'.format(loss.item())
                trained_sum_iters = len(train_loader) * epoch + i_batch + 1
                average_time = (time.time() - start_time) / trained_sum_iters
                remain_time = average_time * (len(train_loader) * 1000 - trained_sum_iters) / 3600
                print('{}\t{}\t lr {:2.3}\t average_time:{:.3f}s\t remain_time:{:.3f}h'.format(Epoch, Loss,
                                                                                               optimizer.param_groups[0][
                                                                                                   'lr'],
                                                                                               average_time,
                                                                                               remain_time))
        scheduler.step()
        # save model
        checkpoint_path = os.path.join("./", 'model_' + str(epoch) + '.pth')

        torch.save(model, checkpoint_path)
