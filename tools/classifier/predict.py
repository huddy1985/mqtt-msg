import torch
from PIL import Image
from torchvision import transforms
import sys
sys.path.append('C:/Users/ilknu/Desktop/DCP ile sis kaldırma ve CNN ile sis tespiti/classifier')

from model import CNNModel  # model dosyasını import et

# Modeli yükle
model = CNNModel()
model.load_state_dict(torch.load('model.pth'))  # Modeli yüklüyoruz
model.eval()  # Değerlendirme moduna alıyoruz

# Görüntü işleme fonksiyonu
def predict(image_path):
    """
    Bu fonksiyon verilen bir görüntü yoluna göre sisli olup olmadığını sınıflandırır.
    """
    image = Image.open(image_path)  # Görüntüyü açıyoruz
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Görüntüyü yeniden boyutlandırıyoruz
        transforms.ToTensor(),  # Görüntüyü tensöre dönüştürüyoruz
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Görüntüyü normalize ediyoruz
    ])
    image = transform(image).unsqueeze(0)  # Modelin kabul edeceği formata dönüştürüyoruz
    output = model(image)  # Modelden tahmin alıyoruz
    prediction = (output.squeeze() > 0.5).float()  # 0.5 eşik değeri ile sınıflandırıyoruz
    return "Hazy" if prediction.item() == 1 else "Clear"  # Sonucu döndürüyoruz
