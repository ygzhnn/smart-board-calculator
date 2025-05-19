import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class CalculatorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Klasör isimlerini oku (etiketler)
        self.labels = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.label_map = {label: idx for idx, label in enumerate(self.labels)}

        # Dosya yolları ve label isimleri
        self.data = []
        for label in self.labels:
            folder_path = os.path.join(root_dir, label)
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.data.append((os.path.join(folder_path, fname), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label_str = self.data[idx]
        image = Image.open(img_path).convert("L")  # Gri tonlama
        if self.transform:
            image = self.transform(image)
        label = self.label_map[label_str]
        return image, label

class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 16 * 16, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)
