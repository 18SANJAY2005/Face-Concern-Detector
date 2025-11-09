import os
from typing import Dict, Tuple, Optional

import numpy as np
import torch

from src.preprocessing import FacePreprocessor
from src.model import FaceConcernDetector
from src.gradcam import GradCAM, get_target_layer
from src.visualization import overlay_heatmap_simple


def predict(model: FaceConcernDetector,
           image_path: str,
           device: str = 'cpu',
           generate_heatmaps: bool = True,
           save_overlay: bool = False,
           output_dir: str = 'outputs') -> Tuple[Dict[str, float], Dict[str, np.ndarray], np.ndarray]:
  
    preprocessor = FacePreprocessor()

    print(f"Processing image: {image_path}")
    face_image, landmarks = preprocessor.detect_and_align_face(image_path)
    original_face = face_image.copy()

    preprocessed = preprocessor.preprocess_for_model(face_image)
    input_tensor = torch.from_numpy(preprocessed).unsqueeze(0).to(device)

 
    model.eval()
    with torch.no_grad():
        scores = model.predict(input_tensor)

    print(f"Detection scores: {scores}")

    heatmaps: Dict[str, np.ndarray] = {}
    if generate_heatmaps:
        try:
            target_layer = get_target_layer(model)
            gradcam = GradCAM(model, target_layer)

            for idx, concern_name in enumerate(model.concern_names):
                cam = gradcam.generate_cam(input_tensor, idx)
                heatmaps[concern_name] = cam

                if save_overlay:
                    os.makedirs(output_dir, exist_ok=True)
                    overlay = overlay_heatmap_simple(original_face, cam)
                    overlay_path = os.path.join(
                        output_dir,
                        f"{os.path.basename(image_path).split('.')[0]}_{concern_name}_overlay.jpg",
                    )
                    import matplotlib.pyplot as plt

                    plt.imsave(overlay_path, overlay)
                    print(f"Saved overlay to {overlay_path}")

        except Exception as e:
            print(f"Warning: Could not generate heat-maps: {e}")
            print("Continuing without heat-maps...")

    return scores, heatmaps, original_face


def predict_from_bytes(model: FaceConcernDetector,
                      image_bytes: bytes,
                      device: str = 'cpu') -> Tuple[Dict[str, float], Optional[str]]:
    
    import io
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    temp_path = 'temp_input.jpg'
    image.save(temp_path)

    try:
        scores, heatmaps, original_face = predict(
            model, temp_path, device, generate_heatmaps=True, save_overlay=True
        )

        overlay_path = 'temp_overlay.jpg'
        if heatmaps:
            combined_heatmap = np.zeros_like(list(heatmaps.values())[0])
            for heatmap in heatmaps.values():
                combined_heatmap = np.maximum(combined_heatmap, heatmap)

            overlay = overlay_heatmap_simple(original_face, combined_heatmap)
            import matplotlib.pyplot as plt

            plt.imsave(overlay_path, overlay)

        return scores, overlay_path

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

