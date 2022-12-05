import torch
from torchvision import datasets, transforms as T
from src.dataset.books import BookDataset
from tqdm import tqdm

transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
dataset = BookDataset('../../../data/book-dataset/book30-listing-train-train.csv', '../../../data/book-dataset/img/',
                      transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

means = []
stds = []
for img, label in tqdm(loader):
    means.append(torch.mean(img, dim=(0, 2, 3)))
    stds.append(torch.std(img, dim=(0, 2, 3)))

mean = torch.mean(torch.stack(means), dim=0)
std = torch.mean(torch.stack(stds), dim=0)

print(mean, std)
