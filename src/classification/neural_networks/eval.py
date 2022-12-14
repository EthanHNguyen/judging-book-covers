"""
This file is used to evaluate the ResNet model on the test set.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from src.classification.neural_networks.dataset.books import BookDataset

img_dir = '../../../data/book-dataset/img/'

torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    # CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Scaler
    scaler = torch.cuda.amp.GradScaler()

    # Hyper-parameters
    batch_size = 256
    epoch = 10

    # Load the dataset
    test_dataset = BookDataset('../../../data/book-dataset/book30-listing-test.csv', img_dir, transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=4)

    # Model
    model = torch.load("models/fcn/epoch_9.pth")
    model.to(device)

    # Test accuracy
    model.eval()
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
    print(f'Test accuracy: {acc:.3f}')