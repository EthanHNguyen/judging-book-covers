import numpy as np
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import cv2

from books import BookDataset

img_dir = '../../../data/book-dataset/img/'

torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    # CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Scaler
    scaler = torch.cuda.amp.GradScaler()

    # Hyper-parameters
    batch_size = 256
    all_epoch = 10
    lr = 1e-1

    # Load the dataset
    train_dataset = BookDataset('../../../data/book-dataset/book30-listing-train-train.csv', img_dir,
                                transform=ToTensor())
    test_dataset = BookDataset('../../../data/book-dataset/book30-listing-train-val.csv', img_dir, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=4)

    # Model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.to(device)

    # Optimizer
    optimizer = SGD(model.parameters(), lr=lr)

    # Loss function
    loss_fn = CrossEntropyLoss()

    for current_epoch in range(all_epoch):
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            train_x = train_x.to(device)
            train_label = train_label.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward + backward + optimize
            predict_y = model(train_x.float())
            with torch.cuda.amp.autocast():
                loss = loss_fn(predict_y, train_label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Print statistics
            # if idx % 10 == 0:
            #     print('idx: {}, loss: {}'.format(idx, loss.sum().item()))

        # Train accuracy
        model.eval()
        all_correct_num = 0
        all_sample_num = 0
        for idx, (test_x, test_label) in enumerate(train_loader):
            test_x = test_x.to(device)

            predict_y = model(test_x.float()).detach().to('cpu')
            predict_y = np.argmax(predict_y, axis=-1)
            current_correct_num = predict_y == test_label
            all_correct_num += np.sum(current_correct_num.numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        acc = all_correct_num / all_sample_num
        print(f'Epoch {current_epoch}. Train accuracy: {acc:.3f}')

        # Validation accuracy
        all_correct_num = 0
        all_sample_num = 0
        for idx, (test_x, test_label) in enumerate(test_loader):
            test_x = test_x.to(device)

            predict_y = model(test_x.float()).detach().to('cpu')
            predict_y = np.argmax(predict_y, axis=-1)
            current_correct_num = predict_y == test_label
            all_correct_num += np.sum(current_correct_num.numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        acc = all_correct_num / all_sample_num
        print(f'Epoch {current_epoch}. Validation accuracy: {acc:.3f}')
        torch.save(model, f'models/resnet-18/epoch_{current_epoch}.pth')
