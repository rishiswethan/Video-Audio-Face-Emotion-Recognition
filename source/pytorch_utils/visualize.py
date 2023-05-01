import cv2
import numpy as np
from PIL import Image as ImagePIL
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def create_gradcam(model, model_input, target_layer, device_name, square_size, verbose=False):
    # Set up hooks to access gradients and feature maps
    gradients = []
    feature_maps = []

    def gradient_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    def feature_map_hook(module, input, output):
        feature_maps.append(output.detach())

    target_layer.register_forward_hook(feature_map_hook)
    target_layer.register_backward_hook(gradient_hook)

    # Forward pass
    model.eval()
    model.to(device_name)
    if verbose:
        print(model_input[0].shape, model_input[1].shape)
        print(type(model_input[0]), type(model_input[1]))
    output = model(torch.from_numpy(np.array(model_input[0])).float().to(device_name),
                   torch.from_numpy(np.array(model_input[1])).float().to(device_name))
    _, pred_class = torch.max(output, 1)

    if verbose:
        print("Grad cam pred class: ", pred_class)
        print("NN output: ", torch.nn.functional.softmax(output, dim=1))

    # Backward pass
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, pred_class] = 1
    output.backward(gradient=one_hot.to(device_name))

    # Compute Grad-CAM
    grad_weighted = torch.mean(gradients[0], dim=[2, 3], keepdim=True)
    grad_cam = F.relu(torch.sum(feature_maps[0] * grad_weighted, dim=1, keepdim=True))
    grad_cam = F.interpolate(grad_cam, size=(square_size, square_size), mode="bilinear", align_corners=False)

    # Normalize the Grad-CAM array to [0, 1]
    grad_cam_np = grad_cam.squeeze().cpu().numpy()
    grad_cam_np = (grad_cam_np - np.min(grad_cam_np)) / (np.max(grad_cam_np) - np.min(grad_cam_np))

    # Convert Grad-CAM to PIL Image
    grad_cam_pil = ImagePIL.fromarray(np.uint8((1.0 - grad_cam_np) * 255))

    return grad_cam_pil


def overlay_gradcam_on_image(img, grad_cam_pil, square_size, alpha=0.5):
    # convert to RGB if needed
    if len(img.shape) == 3:
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.repeat(img, 3, axis=2)
    img = img.astype(np.uint8)

    # Resize the original image to the Grad-CAM image size
    original_img_resized = cv2.resize(img, (square_size, square_size))

    # Apply a colormap to the Grad-CAM image
    grad_cam_np = np.array(grad_cam_pil)
    grad_cam_colored = cv2.applyColorMap(grad_cam_np, cv2.COLORMAP_JET)

    # Overlay Grad-CAM on the original image
    overlaid_image = cv2.addWeighted(original_img_resized, 1 - alpha, grad_cam_colored, alpha, 0)

    result_pil = ImagePIL.fromarray(overlaid_image)

    return result_pil


def visualise_feature_maps(feature_map, feature_map_name):
    feature_map = feature_map.cpu().numpy()

    # Get the number of feature maps
    num_feature_maps = feature_map.shape[1]

    # Calculate the number of rows and columns for the plot
    num_cols = 8
    num_rows = num_feature_maps // num_cols + int(num_feature_maps % num_cols > 0)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    for i in range(num_feature_maps):
        ax = axes[i // num_cols, i % num_cols]
        ax.imshow(feature_map[0, i], cmap="viridis")
        ax.axis("off")

    # Hide empty subplots
    for i in range(num_feature_maps, num_rows * num_cols):
        axes[i // num_cols, i % num_cols].axis("off")

    plt.savefig(feature_map_name)
