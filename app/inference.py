# app/inference.py
import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
from app.models import UNet, MelanomaClassifier  # Import model architectures
from app.utils import circular_crop  # circular crop utility

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
segmentation_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

classification_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def load_models():
    try:
        ROOT_DIR = Path(__file__).resolve().parents[1]
        MODEL_DIR = ROOT_DIR / "trained_weights"

        seg_model = UNet(n_channels=3, n_classes=1, bilinear=False)
        seg_model.load_state_dict(torch.load(MODEL_DIR / "unet_dice0.8369_save1746204838.pt", map_location=device))
        seg_model.to(device).eval()

        clf_model = MelanomaClassifier(input_shape=3, hidden_units=10, output_shape=1).to(device)
        checkpoint = torch.load(MODEL_DIR / "model_with-UNET_epoch16+3+16_acc0.9024_thres0.2_minpixel5000_batch32.pth", map_location=device)
        clf_model.load_state_dict(checkpoint["model_state_dict"])
        clf_model.to(device).eval()

        print("✅ Models loaded successfully.")
        return seg_model, clf_model

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise RuntimeError("Failed to load models") from e


def predict_melanoma(image: Image.Image, seg_model, clf_model, threshold=0.5, min_pixels=5000):
    seg_input = segmentation_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        mask_pred = seg_model(seg_input)
        mask_pred = (mask_pred > threshold).float()

    mask_np = mask_pred.squeeze().cpu().numpy()
    pixel_count = (mask_np > 0).sum()

    if pixel_count < min_pixels:
        processed_image = circular_crop(image)
    else:
        mask_resized = Image.fromarray((mask_np * 255).astype("uint8")).resize(image.size)
        mask_np_resized = np.array(mask_resized) / 255.0
        mask_np_exp = np.expand_dims(mask_np_resized, axis=-1)
        image_np = np.array(image)
        segmented_np = (image_np * mask_np_exp).astype("uint8")
        processed_image = Image.fromarray(segmented_np)

    clf_input = classification_transform(processed_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = clf_model(clf_input)
        prob = torch.sigmoid(output).item()

    return prob, image, mask_resized if pixel_count >= min_pixels else None, processed_image
