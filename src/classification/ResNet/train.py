import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms

from FCN_model import FCN

from src.dataset.books import BookDataset

img_dir = '../../../data/book-dataset/img/'

torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    # CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Scaler
    # scaler = torch.cuda.amp.GradScaler()

    # Hyper-parameters
    batch_size = 256
    all_epoch = 10
    lr = 1e-1
    transforms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5531, 0.5136, 0.4756], std=[0.3287, 0.3141, 0.3136])])

    # Load the dataset
    train_dataset = BookDataset('../../../data/book-dataset/book30-listing-train-train.csv', img_dir,
                                transform=transforms)
    test_dataset = BookDataset('../../../data/book-dataset/book30-listing-train-val.csv', img_dir, transform=transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=4)

    # Model
    model = FCN()
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
            loss = loss_fn(predict_y, train_label)

            loss.backward()
            optimizer.step()

            # Print statistics
            if idx % 10 == 0:
                print('idx: {}, loss: {}'.format(idx, loss.sum().item()))

        # Train accuracy
        # model.eval()
        # all_correct_num = 0
        # all_sample_num = 0
        # for idx, (test_x, test_label) in enumerate(train_loader):
        #     test_x = test_x.to(device)
        #
        #     predict_y = model(test_x.float()).detach().to('cpu')
        #     predict_y = np.argmax(predict_y, axis=-1)
        #     current_correct_num = predict_y == test_label
        #     all_correct_num += np.sum(current_correct_num.numpy(), axis=-1)
        #     all_sample_num += current_correct_num.shape[0]
        # acc = all_correct_num / all_sample_num
        # print(f'Epoch {current_epoch}. Train accuracy: {acc:.3f}')

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
        torch.save(model, f'models/fcn/epoch_{current_epoch}.pth')
