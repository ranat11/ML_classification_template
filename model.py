from torch import nn  

class NumberDetection(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(NumberDetection, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), )
        self.bn2 = nn.BatchNorm2d(16)

        self.fc1_1 = nn.Linear(18000, num_classes)
        self.fc1_2 = nn.Linear(18000, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(x)
        x = self.relu(x)

        x = x.reshape(x.shape[0], -1)
        x1 = self.fc1_1(x)
        x2 = self.fc1_2(x)
        return x1, x2 


if __name__ == '__main__':
    from dataset import CreateDataset
    from utils import check_accuracy
    from torch.utils.data import  DataLoader 
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = CreateDataset(train = True, mnist = False, download = False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=12, shuffle=True)
    
    model = NumberDetection(in_channels=1, num_classes=10).to(device)
    load_model  = torch.load("model/number_crop.pth.tar")
    model.load_state_dict(load_model['state_dict'])

    print(f"Accuracy on test set: {check_accuracy(test_loader, model, device)*100:.2f}")