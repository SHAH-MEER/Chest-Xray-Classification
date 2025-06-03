import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define image transforms 
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalization values used in DenseNet pretrained weights
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load DenseNet121 with pretrained weights
weights = models.DenseNet121_Weights.DEFAULT
model = models.densenet121(weights=weights)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Modify classifier like in your notebook
in_feats = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_feats, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2)
)

# Load saved best model weights
model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
model.to(device)
model.eval()

classes = ['Normal', 'Pneumonia'] 

def predict_image(img):
    # img is a PIL image from Gradio upload
    input_tensor = preprocess(img).unsqueeze(0).to(device) 
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    
    conf, pred_idx = torch.max(probs, dim=1)
    return {classes[pred_idx.item()]: conf.item()}

# Gradio Interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type='pil'),
    outputs=gr.Label(num_top_classes=2),
    title="Pneumonia Detection from X-ray",
    description="Upload a chest X-ray image and the model will predict Normal or Pneumonia with confidence."
)

iface.launch()
