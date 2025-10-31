import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CNNModel  # model.py'yi import et

def evaluate():
    # Veriyi ön işleme
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Test veri seti
    test_dataset = datasets.ImageFolder('C:/Users/ilknu/Desktop/CNN ile Sis Tespiti/data/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Device tanımlaması
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Modeli yükle
    model = CNNModel()
    model.load_state_dict(torch.load('model.pth'))
    model.to(device)  # Modeli uygun cihaza taşı
    model.eval()
    
    # Doğrulama işlemi
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Cihaza taşı
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted.view(-1) == labels).sum().item()
    
    accuracy = correct / total
    accuracy_percentage = accuracy * 100  # Yüzdelik başarı oranı
    print(f"Test Accuracy: {accuracy_percentage:.2f}%")  # Yüzde formatında yazdır

if __name__ == "__main__":
    evaluate()
