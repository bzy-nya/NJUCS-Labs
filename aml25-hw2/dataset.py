import pandas
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

image_size = 28

train_data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.Resize(image_size),
    transforms.ToTensor()
])

eval_data_transform = transforms.Compose([
    transforms.ToPILImage(),  
    transforms.Resize(image_size),
    transforms.ToTensor()
])

class FMDataset(Dataset):
    def __init__(self, image_csv, label_csv=None, transform=None): 
        self.transform = transform
        self.images = image_csv.values.reshape(-1, 28, 28).astype(np.uint8)
        self.labels = label_csv['label'].values if label_csv is not None else None
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx] if self.labels is not None else -1
        
        if self.transform: 
            image = self.transform(image)
        return image, label


X_train, X_valid, y_train, y_valid = train_test_split(
    pandas.read_csv("./data/train_image_labeled.csv"), 
    pandas.read_csv("./data/train_label.csv"),
    test_size=0.2, 
    shuffle=True
)
train_labeled_data = FMDataset(X_train, y_train, transform=train_data_transform)
valid_labeled_data = FMDataset(X_valid, y_valid, transform=eval_data_transform)
train_unlabeled_data = FMDataset(pandas.read_csv("./data/train_image_unlabeled.csv"), transform=train_data_transform)
test_data = FMDataset(pandas.read_csv("./data/test_image.csv"), transform=eval_data_transform)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    train_labeled_loader = DataLoader(train_labeled_data, batch_size=512, shuffle=True)
    
    image, label = next(iter(train_labeled_loader))
    print(image.shape, label.shape)