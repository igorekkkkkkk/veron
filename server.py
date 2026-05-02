from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision
import os
import requests
import gdown

app = FastAPI()

# ✅ CORS (чтобы сайт работал)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📦 путь к модели
MODEL_PATH = "food_model.pth"

# 🔽 скачивание модели если нет

if not os.path.exists(MODEL_PATH):
    print("Скачиваю модель...")
    # Прямая ссылка для gdown (нужен только ID)
    file_id = "1iJlftLIQrfj1PoTwfrzW-fTTXQB8cigw"
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, MODEL_PATH, quiet=False)
else:
    print("Модель уже загружена.")

# 🧠 загрузка модели
model = torchvision.models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 5)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()

# 📚 классы (ВАЖНО: как в train.py)
classes = ['Dessert', 'Fried food', 'Noodles-Pasta', 'Rice', 'Vegetable-Fruit']

# 🔥 калории
calories = {
    "Dessert": 200,
    "Fried food": 295,
    "Noodles-Pasta": 130,
    "Rice": 212,
    "Vegetable-Fruit": 80
}

# 🖼 обработка изображения
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 🚀 API
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    weight: int = Form(...)
):
    image = Image.open(file.file).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)

    food = classes[pred.item()]
    cal = calories[food] * weight / 100

    return {
        "food": food,
        "calories": round(cal, 2)
    }

# 🧪 тестовый маршрут
@app.get("/")
def home():
    return {"message": "Food AI работает 🚀"}
