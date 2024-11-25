import argparse
import torch
import cv2
import numpy as np
from collections import OrderedDict
import segmentation_models_pytorch as smp

def load_model(checkpoint_path):
    model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model'].items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    
    model.eval()
    return model

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.tensor(image, dtype=torch.float).unsqueeze(0)

def postprocess_mask(mask):
    mask = mask.squeeze(0).detach().numpy()
    mask = np.argmax(mask, axis=0)
    return mask

def main(args):
    # Load the trained model
    model = load_model(args.checkpoint_path)
    
    input_image = preprocess_image(args.image_path)
    
    with torch.no_grad():
        output_mask = model(input_image)
    
    mask = postprocess_mask(output_mask)
    
    # Save the mask as an image
    mask = (mask * 85).astype(np.uint8)
    output_path = "segmented_output.png"
    cv2.imwrite(output_path, mask)
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for U-Net segmentation")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--checkpoint_path", type=str, default="model.pth", help="Path to the model checkpoint")
    args = parser.parse_args()
    main(args)
