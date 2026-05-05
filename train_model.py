import torch
import torch.nn as nn
import torch.optim as optim

from models.gesture_net import GestureNet
from utils.train_utils import get_dataloaders, train_one_epoch, validate


# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IMPORTANT: correct dataset path
train_loader, val_loader = get_dataloaders(
    "data/data/gestures",
    batch_size=32
)

model = GestureNet(num_classes=5, dropout=0.4, norm="batch").to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

best_acc = 0
num_epochs = 10

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")

    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device
    )

    val_acc = validate(
        model, val_loader, criterion, device
    )

    print(
        f"Epoch {epoch + 1} Summary -> "
        f"Train Loss: {train_loss:.3f}, Val Acc: {val_acc:.3f}"
    )

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "models/best_model.pth")
        print("✅ Saved new best model!")

print(f"\n🎯 Training complete! Best validation accuracy: {best_acc:.3f}")
