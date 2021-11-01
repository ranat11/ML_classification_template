import os
import random
import cv2
import numpy as np
import string

import torch
import torchvision
import torchvision.transforms as transforms  
from torch.utils.data import Dataset, DataLoader  
from torchvision.utils import make_grid

from utils import AddGaussianNoise

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class CreateDataset(Dataset):
    def __init__(self):
        self.len_data = 100_000
        self.image_dim = (60,60)

        ## Load Data     
        #TODO this git repository does not include the dataset, in order to test it, you need to load from the kaggle and put it in the following folder
        # https://www.kaggle.com/thomasqazwsxedc/alphabet-characters-fonts-dataset?select=Images
        self.al_classes = list(string.ascii_uppercase)
        self.al_dataset_path = "dataset/alphabet_font"
        self.al_class_len = 14990-1

        #TODO this git repository does not include the dataset, in order to test it, you need to load from the kaggle and put it in the following folder
        # https://www.kaggle.com/karnikakapoor/digits
        self.num_classes = [str(i) for i in range(10)]
        self.num_dataset_path = "dataset/digit_font"
        self.num_class_len = 1016-1

    def __len__(self):
        return self.len_data

    def __getitem__(self, index):
        def choosefile(al = True):
            if al:
                path = os.path.join(self.al_dataset_path, self.al_classes[random.randint(0, 25)])
                file_name = os.path.join(path, str(random.randint(0, self.al_class_len)) + ".png")

                img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, self.image_dim)
                img = cv2.bitwise_not(img)

                return img

            else:
                choose_num = random.randint(0, 9)
                path = os.path.join(self.num_dataset_path, str(choose_num))
                file_name = os.path.join(path, "img"+ format(choose_num+1, '03d') + "-" + format(random.randint(1, self.num_class_len), '05d') + ".png")
                img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, self.image_dim)

                return img, choose_num
                
        a1 = choosefile(al=True)
        a2 = choosefile(al=True)
        n1 = choosefile(al=False)
        n2 = choosefile(al=False)
        a3 = choosefile(al=True)
        
        img = np.concatenate((a1, a2, n1[0], n2[0] ,a3), 1)
        target = (n1[1], n2[1])

        img = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomPerspective(distortion_scale=0.4, p=1.0),
            transforms.RandomAdjustSharpness(sharpness_factor=0, p=1.0),
            # transforms.RandomAffine(degrees=(0, 10), translate=(0.1, 0.3), scale=(0.5, 0.75)), 
            AddGaussianNoise(0., 0.7)
            ])(img)
            
        if random.random() > 0.5:
            img = transforms.functional.rotate(img, 180)

        target = torch.as_tensor(target)

        return img, target

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    transform = transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = CreateDataset()
    dataset_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

    for images, labels in dataset_loader:
        print('images.shape:', images.shape)
        plt.figure(figsize=(16,8))
        plt.axis('off')
        plt.imshow(make_grid(images).permute((1, 2, 0)))
        print(labels)
        break
    plt.show()

    images, targets = next(iter(dataset_loader)) 
