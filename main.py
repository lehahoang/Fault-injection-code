from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from dataset.dataloader import mnist_loader
from dataset.dataloader import cifar10_loader
from dataset.dataloader import cifar100_loader
from dl_models.lenet5 import lenet5
from dl_models.alexnet import alexnet
from dl_models.googlenet import googlenet

from phase import train,test

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

def main():
    print("|| Loading.....")
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
    parser.add_argument('--net', type=str, help='Class of DNN model')
    parser.add_argument('--train-batch-size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='Batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for train process (defaul is 10 for mnist_lenet5, 40 for AlexNet, and 200 for Googlenet)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Showing loging status after a certain number of batches ')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--train', action='store_true',
                        help='Performing trainning process')
    parser.add_argument('--test', action='store_true',
                        help='Performing testing/validation process')
    parser.add_argument('--load-file', type=str,
                        help='Load the pre-trained weights')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model_class_map[args.net].to(device)
    train_,test_= dataset_class_map[args.net](args.train_batch_size, args.test_batch_size)

    if args.train:
        print("|| Starting the training phase")
        print("|| DNN model to be trained:", args.net)
        print("|| Dataset to be used:", dataset_class_map[args.net])
        print("|| Number of epochs:", args.epochs)
        if use_cuda: print("|| Training model on GPU\n")
        if args.net=='lenet5':
            optimizer=optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
            for epoch in range(1, args.epochs + 1):
                train(args.log_interval, model, device, train_, optimizer, epoch)
                test(model, device, test_)
            if args.save_model:
                torch.save(model.state_dict(),"./trained_models/mnist_lenet5.pt")


        elif args.net=='alexnet':
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
            for epoch in range(1, args.epochs + 1):
                scheduler.step()
                train(args.log_interval, model, device, train_, optimizer, epoch)
                test(model, device, test_)
            if args.save_model:
                torch.save(model.state_dict(),"./trained_models/cifar10_alexnet.pt")


        elif args.net=='googlenet':
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum =0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
            for epoch in range(1, args.epochs + 1):
                scheduler.step()
                train(args.log_interval,  model, device, train_, optimizer, epoch)
                test(model, device, test_)
            if (args.save_model):
                torch.save(model.state_dict(),"./trained_models/cifar100_googlenet.pt")


    if args.test:
        weight_file=args.load_file
        model.load_state_dict(torch.load(weight_file, map_location='cpu'))
        test(model, device, test_)

if __name__ == '__main__':
    main()
