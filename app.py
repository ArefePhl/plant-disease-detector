"""Simple Streamlit interface for plant-disease prediction."""

import io

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


MODEL_PATH = "weights/resnet18_best.pth"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    class_names = checkpoint["class_names"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, class_names, device


def predict(image, model, class_names, device, top_k=5):
    preprocessing = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    tensor = preprocessing(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    values, indices = probabilities.topk(min(top_k, len(class_names)))
    return [(class_names[i.item()], value.item()) for value, i in zip(values, indices)]


st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")
st.title("🌿 تشخیص بیماری گیاه")
st.caption("مدل ResNet-18 آموزش‌دیده روی ۳۸ کلاس دیتاست PlantVillage")

uploaded = st.file_uploader("تصویر برگ را انتخاب کنید", type=["jpg", "jpeg", "png", "webp"])
if uploaded:
    image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
    st.image(image, caption="تصویر ورودی", use_container_width=True)
    model, class_names, device = load_model()
    predictions = predict(image, model, class_names, device)
    st.subheader("نتایج")
    for rank, (name, confidence) in enumerate(predictions, start=1):
        st.write(f"{rank}. **{name}** — {confidence * 100:.2f}%")
        st.progress(confidence)
    if predictions[0][1] < 0.60:
        st.warning("اعتماد مدل پایین است؛ تصویر واضح‌تر و نزدیک‌تر از برگ تهیه کنید.")
