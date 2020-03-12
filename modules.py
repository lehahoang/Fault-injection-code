import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torchvision import datasets, transforms
import torch.optim as optim
from dataset.dataloader import mnist_loader
from dataset.dataloader import cifar10_loader
from dataset.dataloader import cifar100_loader
from dl_models.lenet5 import lenet5
from dl_models.alexnet import alexnet
from dl_models.googlenet import googlenet
from faultInjection import layer, randNetwork


model_class_map = {
                   'lenet5'          : lenet5(),
                   'alexnet'         : alexnet(),
                   'googlenet'       : googlenet()
                  }
dataset_class_map={
                   'lenet5'          : mnist_loader,
                   'alexnet'         : cifar10_loader,
                   'googlenet'       : cifar100_loader
                  }

def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

def val(model, device, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.sampler),
        100. * correct / len(data_loader.sampler)))

def prepare_model(args):
    '''
        Preparing the model
    '''
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model_class_map[args.net].to(device)
    weight_file=args.load_from
    train_, val_, test_= dataset_class_map[args.net](args, args.train_batch_size, args.test_batch_size)
    return model, train_, val_, test_, device, weight_file

def show_info(weight_file, model):
    '''
        Showing the required info:
        -- DNN models in detail
        -- Name of the layers
    '''
    print(" =================== NETWORK INFO ========================")
    model_params = torch.load(weight_file, map_location='cpu')
    layers_name=list(model_params.keys())
    print(model,'\n')
    print('Name of the layers:', layers_name)
    print('===========================================================')
