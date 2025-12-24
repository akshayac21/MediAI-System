import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import base64
from flask import render_template
import matplotlib.pyplot as plt
from medmnist import INFO

# --- CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./models/octmnist_resnet18.pth"

DATASET_NAME = 'octmnist'
info = INFO[DATASET_NAME]
CLASSES = [info['label'][str(i)] for i in range(len(info['label']))]
NUM_CLASSES = len(CLASSES)

# --- MODEL ---
class ResNet18MedMNIST(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet18 = torchvision.models.resnet18(weights=None)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activation = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def save_activation(self, module, input, output):
        self.activation = output

    def generate_heatmap(self, input_image, class_idx):
        output = self.model(input_image)
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[:, class_idx] = 1
        output.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activation).sum(dim=1)
        cam = F.relu(cam)
        cam = cam.cpu().detach().numpy()
        cam = (cam - cam.min()) / cam.max()
        return cam[0]

# --- LOAD MODEL ---
def get_model():
    model = ResNet18MedMNIST(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    return model

model = get_model()

# --- HELPERS ---
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    return transform(image).unsqueeze(0).to(device), image

def encode_image_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# --- MAIN FUNCTION CALLED FROM app.py ---
def rectina_predict(request):
    file = request.files['file']
    img_bytes = file.read()
    input_tensor, original_img = transform_image(img_bytes)

    output = model(input_tensor)
    probs = F.softmax(output, dim=1)
    idx = torch.argmax(probs).item()

    gradcam = GradCAM(model, model.resnet18.layer4[1].conv2)
    heatmap = gradcam.generate_heatmap(input_tensor, idx)

    heatmap = Image.fromarray(np.uint8(255 * heatmap)).resize(original_img.size)
    heatmap = plt.get_cmap('jet')(np.array(heatmap) / 255.0)[:, :, :3]
    heatmap = np.uint8(heatmap * 255)

    result = Image.fromarray(
        np.uint8(np.array(original_img.convert('RGB')) * 0.6 + heatmap * 0.4)
    )

    return render_template(
        "rectina_index.html",
        prediction=CLASSES[idx],
        confidence=f"{probs[0][idx]*100:.2f}",
        original_img=encode_image_to_base64(original_img.convert('RGB')),
        gradcam_img=encode_image_to_base64(result)
    )
