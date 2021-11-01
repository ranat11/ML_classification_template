
import torch
import torchvision 
from torch import nn 
from torch import optim  

import torchvision.transforms as transforms  
from torch.utils.data import  DataLoader 

from tqdm import tqdm  

from dataset import CreateDataset
from model import NumberDetection
from utils import  check_accuracy


def main(model, continue_training=False):
    # Load dataset
    dataset = CreateDataset()
    
    indices = torch.randperm(len(dataset)).tolist()
    train_dataset = torch.utils.data.Subset(dataset, indices[:-int(len(dataset)*0.2)])
    test_dataset = torch.utils.data.Subset(dataset, indices[-int(len(dataset)*0.2):])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

        
    # Initialize network
    model = NumberDetection(in_channels=in_channels, num_classes=num_classes).to(device)

   
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2, verbose=True)

    # Load weight
    if continue_training:
        load_model  = torch.load("model/position_detection.pth")
        model.load_state_dict(load_model['state_dict'])
        optimizer.load_state_dict(load_model['optimizer'])
    
    # Train Network
    for epoch in range(num_epochs):
        losses = []
        loop =  tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

        for batch_idx, (data, targets) in loop:
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            out1, out2  = model(data)
            loss1 = criterion(out1, targets[:,0])
            loss2 = criterion(out2, targets[:,1])
            loss = loss1+loss2

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            # update progress bar
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss = loss.item())
            losses.append(loss.item())

        mean_loss = sum(losses) / len(losses)
        scheduler.step(mean_loss)

    print("done training")

    print("checking accuracy")
    # print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
    print(f"Accuracy on test set: {check_accuracy(test_loader, model, device)*100:.2f}")

    return model, optimizer


if __name__ == '__main__':
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = 1 
    num_classes = 10

    # Hyperparameters
    learning_rate = 0.0002
    batch_size = 64
    num_epochs = 5

    model = NumberDetection(in_channels=in_channels, num_classes=num_classes).to(device)
    model, optimizer = main(model, continue_training = False)

    print("saving model")
    save_model = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(save_model, "model/position_detection.pth")