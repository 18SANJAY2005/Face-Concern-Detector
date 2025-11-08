import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, List


class GradCAM:
 
    
    def __init__(self, model, target_layer):
      
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor: torch.Tensor, concern_idx: int) -> np.ndarray:
      
        self.model.eval()
        
       
        output = self.model(input_tensor)
        
    
        self.model.zero_grad()
        
     
        target_score = output[0, concern_idx]
        target_score.backward()
        
        
        gradients = self.gradients[0]  
        activations = self.activations[0]  
        
      
        weights = torch.mean(gradients, dim=(1, 2))  
        
        
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
      
        cam = F.relu(cam)
        
      
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
       
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        
        return cam
    
    def overlay_heatmap(self, original_image: np.ndarray, cam: np.ndarray, 
                       alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
      
        if cam.shape != original_image.shape[:2]:
            cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        
     
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
   
        overlay = (alpha * heatmap + (1 - alpha) * original_image).astype(np.uint8)
        
        return overlay


def get_target_layer(model):
   
    backbone = model.backbone
    
 
    resnet_model = None
    
 
    for child in backbone.children():
        if hasattr(child, 'layer4'):
            resnet_model = child
            break

    if resnet_model and hasattr(resnet_model, 'layer4'):
       
        layer4 = resnet_model.layer4
        if len(layer4) > 0:
            last_block = layer4[-1]
            if hasattr(last_block, 'conv2'):
                return last_block.conv2
    

    last_conv = None
    for module in backbone.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    
    if last_conv is not None:
        return last_conv
    
    raise ValueError("Could not find target layer for Grad-CAM")

