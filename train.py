import torch
from tqdm import tqdm
from torch.optim import Adam, lr_scheduler
from models import IMCLNet
from utils import MalwareImageDataset, chooseDataset, get_parameter_number
from torch.utils.data import DataLoader, random_split
import numpy as np
import argparse
from torch import nn
import time
from sklearn.model_selection import KFold


criterion = nn.CrossEntropyLoss()

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()

    with torch.no_grad():
        for _,(x, y)in tqdm(enumerate(test_loader),total =len(test_loader), leave = True):
            x, y = x.cuda(), y.cuda()

            outputs = model(x)
            test_loss += criterion(outputs, y).item() * x.size(0)

            y_pred = torch.argmax(outputs, dim=1)
            correct += (y_pred == y).sum().float()
            total += len(y)

        test_loss /= total
        test_acc = correct / total
        return test_loss, test_acc

def train(fold, model, train_loader, test_loader, args):

    t0 = time.time()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(args.epochs):
        model.train()
        ti = time.time()
        training_loss = 0.0
        train_acc = 0
        correct = torch.zeros(1).squeeze().cuda()
        total = torch.zeros(1).squeeze().cuda()
        for _,(x, y)in tqdm(enumerate(train_loader),total =len(train_loader), leave = True):
            x, y = x.cuda(), y.cuda()
            outputs = model(x)  

            loss = criterion(outputs, y)
            y_pred = torch.argmax(outputs, dim=1)
            correct += (y_pred == y).sum().float()
            total += len(y)

            training_loss += loss.item() * x.size(0) 

            # backward
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 

        scheduler.step()

        train_loss = training_loss / total
        train_acc = correct / total
        test_loss, test_acc = test(model, test_loader)

        print('Epoch: {:03d}'.format(epoch),
                'train_loss: {:.5f}'.format(train_loss),
                'train_acc: {:.3f}%'.format(train_acc * 100),
                'test_loss: {:.5f}'.format(test_loss),
                'test_acc: {:.3f}%'.format(test_acc * 100),
                'time: {:0.2f}s'.format(time.time() - ti))

    dateStr = time.strftime("%d-%b-%Y-%H:%M:%S", time.localtime())

    model_dir = args.models_dir + args.dataset + '/all-resized-' + str(args.image_size) + '/' + 'fold-' + str(fold + 1) + '-' + dateStr + '.pt'

    torch.save(model.state_dict(), model_dir)
    print('Trained model saved to %s.pt' % dateStr)
    print("Total time = %ds" % (time.time() - t0))
    return test_acc


def training(args):

    # load data
    dataset_dir = args.data_dir + args.dataset + '/all-resized-' + str(args.image_size) + '/'
    dataset_name = args.dataset
    
    dataset = MalwareImageDataset(dataset_dir, dataset_name)
    labels = chooseDataset(dataset_name)

    acc = []

    if args.kfold == 0:
        # split datasetSS
        length = len(dataset)

        train_size = int(0.75 * length)       
        test_size = length - train_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # define model
        model = IMCLNet(class_num=len(labels))
        model.cuda()
        total_num, trainable_num = get_parameter_number(model)
        print("'Total': {}, 'Trainable': {}".format(total_num, trainable_num))

        final_test_acc = train(args.kfold, model, train_dataloader, test_dataloader, args)
        acc.append(final_test_acc.item())

    else:
        kfold = KFold(n_splits=args.kfold, shuffle=True)
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            # dataloader
            train_dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_subsampler)
            test_dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=test_subsampler)

            # define model
            model = IMCLNet(class_num=len(labels))
            model.cuda()
            total_num, trainable_num = get_parameter_number(model)
            print("'Total': {}, 'Trainable': {}".format(total_num, trainable_num))

            final_test_acc = train(fold, model, train_dataloader, test_dataloader, args)
            acc.append(final_test_acc.item())

    Average = round(np.mean(acc), 5)   
    print('Average:', Average)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Malware Image")
    parser.add_argument('--dataset', default='MalImg', help='MalImg or BIG2015')
    parser.add_argument('--data_dir', default='./datasets/') 
    parser.add_argument('--image_size', default=32, type=int, help='Image size') 
    parser.add_argument('--kfold', default=5, help='only 0 or 5') 
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--lr', default=0.0038)
    parser.add_argument('--epochs', default=50)
    parser.add_argument('--result_dir', default='./results/')
    parser.add_argument('--models_dir', default='./models/')
    args = parser.parse_args()

    training(args=args)