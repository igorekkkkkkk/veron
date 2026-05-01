from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Form


app = FastAPI()

# загрузка модели
model = torchvision.models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 5)  # поменяй если классов больше
model.load_state_dict(torch.load("food_model.pth"))
model.eval()

classes = ['Dessert', 'Fried food', 'Noodles-Pasta', 'Rice', 'Vegetable-Fruit']  # ВАЖНО: как в train.py

calories = {
    "Dessert": 200,
    "Fried food": 295,
    "Noodles-Pasta": 130,
    "Rice": 212,
    "Vegetable-Fruit": 80
}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    weight: int = Form(...)
):
    image = Image.open(file.file)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)

    food = classes[pred.item()]
    cal = calories[food] * weight / 100

    return {
        "food": food,
        "calories": cal
    }