import os
import cv2
import torch
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet

# RealESRGAN-compatible lightweight enhancer (works with NumPy 1.x)
def enhance_image(image):
    # Load RRDBNet (Real-ESRGAN x4)
    model_path = os.path.join("models", "RealESRGAN_x4plus.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Missing RealESRGAN_x4plus.pth in models/")

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    with torch.no_grad():
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        out = model(img).clamp(0, 1)
        out_img = (out.squeeze(0).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    return out_img
