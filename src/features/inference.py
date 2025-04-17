import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import yaml
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from torchvision.transforms import functional as F


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(model_cfg_data, model_path):
    """Load model with exact weight initialization and verification"""
    # Initialize model
    model = models.resnet50(weights = None)
    num_classes = model_cfg_data.get('num_class', 15)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if model_path:
        try:
            print(f"Loading model from: {model_path}")
            # First try with proper safety measures
            try:
                with torch.serialization.safe_globals([
                    transforms.Compose,
                    transforms.Resize,
                    transforms.CenterCrop,
                    transforms.ToTensor,
                    transforms.Normalize,
                    F.InterpolationMode
                ]):
                    checkpoint = torch.load(model_path, map_location = 'cpu', weights_only = True)
            except Exception as e:
                print(f"Secure loading failed: {str(e)}")
                print("Falling back to weights_only=False")
                checkpoint = torch.load(model_path, map_location = 'cpu', weights_only = False)

            # # Print model structure before loading weights
            # print(f"Original model structure:")
            # print(model)

            # # Print checkpoint format for debugging
            # print(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dictionary'}")

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # # Print a few keys from state dict to verify format
            # print("Sample keys from state_dict:")
            # for i, k in enumerate(list(state_dict.keys())[:5]):
            #     print(f"  {k}")

            # Remove 'model.' prefix if present in keys
            new_state_dict = {}
            for k, v in state_dict.items():
                # Try different potential key mappings
                if k.startswith('model.'):
                    new_state_dict[k.replace('model.', '')] = v
                else:
                    new_state_dict[k] = v

            # Load state dict and capture missing and unexpected keys
            incompatible_keys = model.load_state_dict(new_state_dict, strict = False)
            if incompatible_keys.missing_keys:
                print(f"Missing keys: {incompatible_keys.missing_keys}")
            if incompatible_keys.unexpected_keys:
                print(f"Unexpected keys: {incompatible_keys.unexpected_keys}")

            # # Ensure the model is in evaluation mode
            # model.eval()
            #
            # # Print model structure after loading weights
            # print(f"Final model structure:")
            # print(model)

            # Get class names if available
            class_names = None
            if isinstance(checkpoint, dict):
                class_names = checkpoint.get('class_names', None)
                if class_names:
                    print(f"Found class names in checkpoint: {class_names}")

            return model, class_names

        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            raise

    # Default return if no model path
    model.eval()

    return model, None


def preprocess_image(image_path, img_size, mean, std):
    """Match preprocessing exactly to Colab environment"""
    # Match the preprocessing in your Colab notebook exactly
    transform = transforms.Compose([
        transforms.Resize(256),  # Use identical resize value
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ])

    # Load image with identical processing
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        tensor = transform(img).unsqueeze(0)

    # Print tensor stats for debugging
    print(f"Image tensor shape: {tensor.shape}")
    print(f"Tensor range: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}")

    return tensor


def predict(image_tensor, model, class_names, threshold = 0.1):
    """Return dictionary of all predictions above threshold"""
    # Always ensure model is in evaluation mode
    model.eval()

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

        print("\nRaw prediction probabilities:")
        for i, (name, prob) in enumerate(zip(class_names, probs)):
            print(f"{i}: {name:20}: {prob:.4f}")

        # Create result dictionary
        results = {}
        for i, (class_name, prob) in enumerate(zip(class_names, probs)):
            if prob >= threshold:
                results[class_name] = float(prob)

        # If nothing above a threshold, return top prediction
        if not results:
            top_idx = np.argmax(probs)
            results[class_names[top_idx]] = float(probs[top_idx])

        return results


def generate_gradcam(image_tensor, model, class_id):
    # Target the last layer of layer4 in ResNet50
    target_layers = [model.layer4[-1]]
    targets = [ClassifierOutputTarget(class_id)]

    # Ensure model allows gradient computation for GradCAM
    model.eval()
    for param in model.parameters():
        param.requires_grad = False  # Start with all frozen

    # Only enable gradients for the target layer
    for param in target_layers[0].parameters():
        param.requires_grad = True

    with torch.set_grad_enabled(True):
        with GradCAM(model = model, target_layers = target_layers) as cam:
            grayscale_cams = cam(input_tensor = image_tensor, targets = targets)
            rgb_img = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            # Normalize image to 0-1 range for visualization
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
            return show_cam_on_image(rgb_img, grayscale_cams[0, :], use_rgb = True)


# ************************************

def colab_style_inference(image_path, model, class_names):
    """Replicates the exact Colab inference pipeline to verify model behavior"""
    # Use exact transforms from your Colab notebook
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    # Load and transform the image, exactly as in Colab
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        image_tensor = transform(img).unsqueeze(0)

    # Predict with identical code
    model.eval()  # Ensure evaluation mode
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    # Print detailed results, exactly as in Colab
    print("\nColab-style Detailed Predictions:")
    sorted_indices = np.argsort(probs)[::-1]  # Sort in descending order
    for idx in sorted_indices:
        name = class_names[idx]
        prob = probs[idx]
        print(f"{name:20}: {prob:.4f}")

    # Return the raw probabilities
    return probs
