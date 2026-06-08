"""
visualize_0602_0605.py
Visualize depth maps for 0602, 0605 models
Usage: python visualize_0602_0605.py
"""
import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import networks
from utils import readlines
import datasets

# ── Settings ──────────────────────────────────────────────
DATA_PATH  = "/home/kty2/Lite-mono_code/kitti_data/"
SPLITS_DIR = "splits/eigen"
SAVE_DIR   = "visualize_output"
NUM_IMAGES = 2

MODELS = {
    "0602": "logs/sqldepth_multiRes_0602/models/weights_16",
    "0605": "logs/sqldepth_multiRes_0605/models/weights_16",
}

# Hyperparameters (common for 0602, 0605)
MODEL_DIM   = 32
NUM_LAYERS  = 50
NUM_FEAT    = 256
PATCH_SIZE  = 20
DIM_OUT     = 128
QUERY_NUMS  = 128
HEIGHT      = 320
WIDTH       = 1024
HEIGHT_LOW  = 192
WIDTH_LOW   = 640
MIN_DEPTH   = 0.01
MAX_DEPTH   = 80.0

os.makedirs(SAVE_DIR, exist_ok=True)

def load_model(weights_path):
    encoder_high = networks.ResnetEncoderDecoder(
        num_layers=NUM_LAYERS, num_features=NUM_FEAT, model_dim=MODEL_DIM)
    encoder_low = networks.ResnetEncoderDecoder(
        num_layers=NUM_LAYERS, num_features=NUM_FEAT, model_dim=MODEL_DIM)
    fusion = networks.MultiResFusion(in_channels=MODEL_DIM)
    depth_decoder = networks.Lite_Depth_Decoder_QueryTr(
        in_channels=MODEL_DIM, patch_size=PATCH_SIZE,
        dim_out=DIM_OUT, embedding_dim=MODEL_DIM,
        query_nums=QUERY_NUMS, num_heads=4,
        min_val=MIN_DEPTH, max_val=MAX_DEPTH)

    encoder_high.load_state_dict(
        torch.load(os.path.join(weights_path, "encoder_high.pth")), strict=False)
    encoder_low.load_state_dict(
        torch.load(os.path.join(weights_path, "encoder_low.pth")), strict=False)
    fusion.load_state_dict(
        torch.load(os.path.join(weights_path, "fusion.pth")), strict=False)
    depth_decoder.load_state_dict(
        torch.load(os.path.join(weights_path, "depth.pth")), strict=False)

    for m in [encoder_high, encoder_low, fusion, depth_decoder]:
        m.cuda()
        m.eval()

    return encoder_high, encoder_low, fusion, depth_decoder

def disp_to_color(disp):
    """Convert disparity map to color image"""
    disp_np = disp.squeeze().cpu().numpy()
    disp_np = (disp_np - disp_np.min()) / (disp_np.max() - disp_np.min() + 1e-8)
    disp_np = (disp_np * 255).astype(np.uint8)
    return cv2.applyColorMap(disp_np, cv2.COLORMAP_MAGMA)

# Load test images
filenames = readlines(os.path.join(SPLITS_DIR, "test_files.txt"))[:NUM_IMAGES]
dataset = datasets.KITTIRAWDataset(
    DATA_PATH, filenames, HEIGHT, WIDTH, [0], 1, is_train=False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Save input images
for idx, inputs in enumerate(dataloader):
    if idx >= NUM_IMAGES:
        break
    img = inputs[("color", 0, 0)].squeeze().cpu().numpy().transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(SAVE_DIR, f"input_img{idx}.png"), img_bgr)
    print(f"Input image {idx} saved")

# Process each model
for model_name, weights_path in MODELS.items():
    print(f"\n=== {model_name} ===")
    encoder_high, encoder_low, fusion, depth_decoder = load_model(weights_path)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for idx, inputs in enumerate(dataloader):
        if idx >= NUM_IMAGES:
            break

        input_color = inputs[("color", 0, 0)].cuda()

        with torch.no_grad():
            High_feature = encoder_high(input_color)
            Low_resol = F.interpolate(input_color, size=(HEIGHT_LOW, WIDTH_LOW),
                                      mode='bilinear', align_corners=False)
            Low_feature = encoder_low(Low_resol)
            Merged_feature = fusion(High_feature, Low_feature)

            # Merged depth
            output_merged = depth_decoder(Merged_feature)
            color_merged = disp_to_color(output_merged[("disp", 0)])
            cv2.imwrite(os.path.join(SAVE_DIR, f"{model_name}_img{idx}_merged.png"), color_merged)

            # For 0605 only: save high/low separately
            if model_name == "0605":
                output_high = depth_decoder(High_feature)
                color_high = disp_to_color(output_high[("disp", 0)])
                cv2.imwrite(os.path.join(SAVE_DIR, f"{model_name}_img{idx}_high.png"), color_high)

                Low_feature_up = F.interpolate(
                    Low_feature, size=High_feature.shape[2:],
                    mode='bilinear', align_corners=False)
                output_low = depth_decoder(Low_feature_up)
                color_low = disp_to_color(output_low[("disp", 0)])
                cv2.imwrite(os.path.join(SAVE_DIR, f"{model_name}_img{idx}_low.png"), color_low)

        print(f"  img{idx} saved")

print(f"\nAll results saved to: {SAVE_DIR}/")