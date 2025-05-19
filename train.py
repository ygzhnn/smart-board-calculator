import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import wandb
import json
from model import SimpleCNN, CalculatorDataset
from tqdm import tqdm

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model():
    DATA_DIR = "/content/CompleteImages/CompleteImages/All data (Compressed)"
    BATCH_SIZE = 32
    EPOCHS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = CalculatorDataset(DATA_DIR, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(dataset.label_map)
    model = SimpleCNN(num_classes).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    early_stopping = EarlyStopping(patience=5)

    wandb.init(project="calculator-recognition", config={
        "learning_rate": 0.001,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "architecture": "SimpleCNN (64x64)"
    })

    print(f"📢 Eğitim başlıyor... Sınıf sayısı: {num_classes}")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for images, labels in tqdm(train_loader, desc=f"📦 Training Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"🔎 Validation Epoch {epoch+1}/{EPOCHS}"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"📊 Epoch {epoch+1} — Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("🛑 Early stopping devreye girdi.")
            break

    torch.save(model.state_dict(), "calculator_model.pth")
    with open("label_map.json", "w") as f:
        json.dump(dataset.label_map, f)

    wandb.save("calculator_model.pth")
    wandb.save("label_map.json")

    print("✅ Eğitim tamamlandı ve model kaydedildi.")
    wandb.finish()

if __name__ == "__main__":
    train_model()
