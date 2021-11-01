import torch

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, prediction1 = scores[0].max(1)
            _, prediction2 = scores[1].max(1)

            prediction1 = torch.unsqueeze(prediction1, 1)
            prediction2 = torch.unsqueeze(prediction2, 1)

            predictions = torch.cat( (prediction1, prediction2), 1)

            num_correct += (predictions == y).sum()/2
            num_samples += predictions.size(0)

    model.train()
    return num_correct/num_samples
