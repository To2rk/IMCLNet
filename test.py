import os
import time
import torch
import argparse
from models import IMCLNet
from torch.utils.data import DataLoader, random_split
from utils import MalwareImageDataset, chooseDataset, ConfusionMatrix, get_parameter_number
from sklearn import metrics
from sklearn.model_selection import KFold


def get_parser():

    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--kfold', default=0, help='only 0 or 5')
    parser.add_argument('--target_fold', default=0, help='select cross validation')
    parser.add_argument('--dataset', default='MalImg', help='MalImg or BIG2015')
    parser.add_argument('--data_dir', default='./datasets/') 
    parser.add_argument('--models_dir', default='./models/')
    parser.add_argument('--image_size', default=32, type=int, help='Image size') 
    parser.add_argument('--test_batch_size', default=16, help='testing batch size.')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--weights', default='fold-0-18-Jul-2020-00:43:44.pt')
    args = parser.parse_args()

    return args


def test(args):

    # load data
    dataset_dir = args.data_dir + args.dataset + '/all-resized-' + str(args.image_size) + '/'
    dataset = args.dataset
    dataset = MalwareImageDataset(dataset_dir, dataset)

    labels = chooseDataset(args.dataset)

    if args.kfold == 0:
    
        # split dataset
        length = len(dataset)

        train_size = int(0.75 * length)          #
        test_size = length - train_size

        _, test_dataset = random_split(dataset, [train_size, test_size])

        # dataloader
        test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    else:
        kfold = KFold(n_splits=args.kfold, shuffle=True)
        for fold, (_, test_ids) in enumerate(kfold.split(dataset)):

            if fold + 1 == args.target_fold:
                test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

                # dataloader
                test_dataloader = DataLoader(dataset, batch_size=args.test_batch_size, sampler=test_subsampler)
        
    model = IMCLNet(class_num=len(labels))
    device = torch.device("cuda:0" if args.cuda else "cpu")
    model.to(device)
    print("Successful to load network!")

    # total_num, trainable_num = get_parameter_number(model)
    # print("'Total': {}, 'Trainable': {}".format(total_num, trainable_num))

    confusion = ConfusionMatrix(num_classes=len(labels), labels_list=labels)

    # load pretrained model
    if args.weights:

        if args.kfold == 0:
            model.load_state_dict(torch.load(args.weights))
        else:
            models_path = args.models_dir + args.dataset  + '/' + 'all-resized-' + str(args.image_size) + '/' + args.weights
            model.load_state_dict(torch.load(models_path))

        print("load pretrained model successful!")
    else:
        print("[Error] Can't found pretrained mode, please check!")
        os._exit()

    # all labels
    out_true = []
    out_pred = []

    model.eval()
    with torch.no_grad():

        correct = torch.zeros(1).squeeze().to(device)
        total = torch.zeros(1).squeeze().to(device)     

        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)

            y_pred = torch.argmax(outputs, dim=1)


            for i, j in zip(y, y_pred):

               confusion.update(j.item(), i.item())

               out_true.append(i.item())
               out_pred.append(j.item())

            correct += (y_pred == y).sum().float()
            total += len(y)

        val_acc = correct / total
    
    confusion.summary()
    confusion.plot('./results/' + args.dataset + '-ConfusionMatrix.png')

    return val_acc.item(), total.item()


if __name__ == "__main__":

    t1 = time.time()

    args = get_parser()
    acc, total = test(args)
    
    t2 = time.time()

    print('Accuracy = {}'.format(acc))

    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / total, total))

