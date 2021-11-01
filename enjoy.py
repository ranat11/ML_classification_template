import torch
import torchvision.transforms as transforms  

import cv2
import os
import time
from model import NumberDetection


def main(path, display = False):        
    load_model  = torch.load("model/position_detection.pth", map_location=device)
    model = NumberDetection(in_channels=in_channels, num_classes=num_classes).to(device)
    model.load_state_dict(load_model['state_dict'])

    for param in model.parameters():
        param.requires_grad = False

    start = time.time()

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name) # shape = (160 * 59)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
        if img.shape[0] > img.shape[1]: img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) 

        # img = cv2.addWeighted( img, 1.5, img, 0, 1.5)
        img = cv2.resize(img, (300,60))

        if display:            
            while True:
                cv2.imshow('image',img)
                if cv2.waitKey(10) & 0xFF == 27:
                    break

            cv2.destroyAllWindows()
            # exit()


        img = transforms.ToTensor()(img)
        img = img[None]
        img = img.to(device=device)

        with torch.no_grad():
            model.eval()

            scores = model(img)
            _, prediction1 = scores[0].max(1)
            _, prediction2 = scores[1].max(1)
            predictions = (int(prediction1), int(prediction2))

        print(f"file: {img_name}, predictions: {predictions}")

    stop=time.time()

    print(f"duration: {stop-start}")

if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # using CPU only
    device = torch.device("cpu")

    in_channels = 1 
    num_classes = 10

    path = "dataset/test_dataset"
    main(path, display = False)

