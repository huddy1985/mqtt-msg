import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CNNModel  # model.py dosyasından CNNModel sınıfını import ediyoruz

# Veri ön işleme (transformasyonlar)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Görüntülerin normalize edilmesi
])

# Eğitim ve test veri setlerinin yüklenmesi
train_dataset = datasets.ImageFolder(r'C:\Users\ilknu\Desktop\DCP ile sis kaldırma ve CNN ile sis tespiti\classifier\data\train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.ImageFolder(r'C:\Users\ilknu\Desktop\DCP ile sis kaldırma ve CNN ile sis tespiti\classifier\data\test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Modeli oluştur
model = CNNModel()

# Kayıp fonksiyonu ve optimizer
criterion = nn.BCELoss()  # İkili sınıflandırma için binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Eğitim döngüsü
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Modeli eğitim moduna al
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Gradients'ı sıfırla
        outputs = model(inputs)  # Modeli çalıştır
        loss = criterion(outputs.squeeze(), labels.float())  # Kayıp hesapla
        loss.backward()  # Backpropagation
        optimizer.step()  # Adımla
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Modeli kaydet
torch.save(model.state_dict(), 'model.pth')
print("Model saved as 'model.pth'")

# Modeli test et
model.eval()  # Modeli test moduna al
correct = 0
total = 0
with torch.no_grad():  # Testte gradient hesaplanmaz
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = (outputs.squeeze() > 0.5).float()  # 0.5 eşik değeri ile sınıflandırma yap
        total += labels.size(0)
        correct += (predicted == labels.float()).sum().item()  # Etiketleri float'a dönüştür

print(f'Test Accuracy: {100 * correct / total:.2f}%')
