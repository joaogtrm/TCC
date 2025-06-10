import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader

def trans_data(train_data_path, test_data_path,S_SIZE,B_SIZE):
    torch.manual_seed(42)
    # All your transform and dataloader logic is here...
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    train_full = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform_train)
    test_full = torchvision.datasets.ImageFolder(root=test_data_path, transform=transform_test)
    
    sample_size = S_SIZE
    
    if sample_size > len(train_full):
        raise ValueError("Sample size cannot be larger than the dataset size.")
    
    indices = torch.randperm(len(train_full))[:sample_size]
    
    train_sample_dataset = Subset(train_full, indices)
    
    trainloader = DataLoader(train_sample_dataset, batch_size=B_SIZE, shuffle=True, num_workers=2)
    testloader = DataLoader(test_full, batch_size=B_SIZE, shuffle=False, num_workers=2)
    
    # ✅ This line MUST be indented to be inside the function
    return (trainloader, testloader)