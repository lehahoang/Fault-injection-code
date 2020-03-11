from __future__ import print_function
import argparse
import torch
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torchsummary import summary
from dataset.dataloader import mnist_loader
from dataset.dataloader import cifar10_loader
from dataset.dataloader import cifar100_loader
from dl_models.lenet5 import lenet5
from dl_models.alexnet import alexnet
from dl_models.googlenet import googlenet
from faultInjection import layer, randNetwork
from phase import train,test,val

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

def main():
    print("|| Loading.....")
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
    parser.add_argument('--net', type=str, help='Class of DNN model')
    parser.add_argument('--train-batch-size', type=int, default=256,
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
    parser.add_argument('--load-from', type=str,
                        help='Load the pre-trained weights')
    parser.add_argument('--out-dir', type=str, default='./trained_models',
                        help='output directory to save the model')
    parser.add_argument('--per-layer-injection', type=str,
                        help='perform layer-wise injection on the specified layer')
    parser.add_argument('--network-injection', action='store_true', default=False,
                    help='perform random fault injection into the network')
    parser.add_argument('--fault-rate', type=float, default=0,
                        help='Fault rate (default:0)')
    parser.add_argument('--show-info', action='store_true', default=False,
                    help='Show model info')
    parser.add_argument('--layer-name', type=str,
                    help='perform layer-wise injection on the specified layer')
    parser.add_argument('--split-fraction', type=float, default =0.1,
                        help='fraction between train/validation set')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)

    model, train_set, val_set, test_set, device, weight_file = prepare_model(args)

    if args.train:
        print("|| Starting the training phase")
        print("|| DNN model to be trained:", args.net)
        print("|| Dataset to be used:", dataset_class_map[args.net])
        print("|| Number of epochs:", args.epochs)
        if use_cuda: print("|| Training model on GPU\n")
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        if args.net=='lenet5':
            optimizer=optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
            for epoch in range(1, args.epochs + 1):
                train(args.log_interval, model, device, train_set, optimizer, epoch)
                test(model, device, val_set)
            torch.save(model.state_dict(), args.out_dir+"mnist_lenet5.pt")


        elif args.net=='alexnet':
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
            for epoch in range(1, args.epochs + 1):
                scheduler.step()
                train(args.log_interval, model, device, train_set, optimizer, epoch)
                test(model, device, val_set)
            torch.save(model.state_dict(), args.out_dir+"cifar10_alexnet.pt")

        elif args.net=='googlenet':
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum =0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
            for epoch in range(1, args.epochs + 1):
                scheduler.step()
                train(args.log_interval,  model, device, train_set, optimizer, epoch)
                test(model, device, val_set)
            torch.save(model.state_dict(), args.out_dir+"cifar100_googlenet.pt")
    if args.test:
        '''
            Doing image classification on the test dataset in clean case
        '''
        model.load_state_dict(torch.load(weight_file, map_location='cpu'))
        # if args.show_info:
        test(model, device, test_set)
    if args.per_layer_injection:
        '''
            Perform fault injection to the specified layer
        '''
        print("|| Performing layer-wise fault injection on the layer", args.per_layer_injection)
        model_params = torch.load(weight_file, map_location='cpu')
        model_params_mutated = layer(args, args.fault_rate, model_params, args.seed, args.per_layer_injection)
        model.load_state_dict(model_params_mutated)
        test(model, device, test_set)

    if args.network_injection:
        '''
            Perform fault injection across the whole network
        '''
        model_params = torch.load(weight_file, map_location='cpu')
        model_params_mutated = randNetwork(args, args.fault_rate, model_params, args.seed)
        model.load_state_dict(model_params_mutated)
        test(model, device, test_set)
        # print('Performing')
    if args.show_info:
        show_info(weight_file, model)

if __name__ == '__main__':
    main()
