import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import Dict, List, Optional
import seaborn as sns


def plot_concern_results(original_image: np.ndarray, 
                        scores: Dict[str, float],
                        heatmaps: Optional[Dict[str, np.ndarray]] = None,
                        save_path: Optional[str] = None) -> None:
   
    num_concerns = len(scores)
    has_heatmaps = heatmaps is not None and len(heatmaps) > 0
    
   
    if has_heatmaps:
     
        fig = plt.figure(figsize=(16, 4 + 3 * num_concerns))
        gs = fig.add_gridspec(2 + num_concerns, 3, hspace=0.3, wspace=0.3)
    else:
        
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_image)
    ax1.set_title('Original Face', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
   
    ax2 = fig.add_subplot(gs[0, 1:])
    concerns = list(scores.keys())
    values = [scores[c] for c in concerns]
    colors = sns.color_palette("husl", len(concerns))
    bars = ax2.barh(concerns, values, color=colors)
    ax2.set_xlabel('Confidence (%)', fontsize=12)
    ax2.set_title('Concern Detection Scores', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)
    
    
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax2.text(val + 1, i, f'{val:.1f}%', 
                va='center', fontsize=10, fontweight='bold')
    
    
    if has_heatmaps:
        for idx, (concern, heatmap) in enumerate(heatmaps.items()):
            row = 1 + idx
            
            
            ax_hm = fig.add_subplot(gs[row, 0])
            ax_hm.imshow(heatmap, cmap='jet')
            ax_hm.set_title(f'{concern.replace("_", " ").title()} Heat-map', 
                          fontsize=12, fontweight='bold')
            ax_hm.axis('off')
            
            
            overlay = overlay_heatmap_simple(original_image, heatmap)
            ax_overlay = fig.add_subplot(gs[row, 1:])
            ax_overlay.imshow(overlay)
            ax_overlay.set_title(f'{concern.replace("_", " ").title()} Overlay '
                               f'({scores.get(concern, 0):.1f}%)', 
                               fontsize=12, fontweight='bold')
            ax_overlay.axis('off')
    
    plt.suptitle('Face Concern Detection Results', fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def overlay_heatmap_simple(original_image: np.ndarray, 
                          heatmap: np.ndarray,
                          alpha: float = 0.4) -> np.ndarray:
  
  
    if heatmap.shape != original_image.shape[:2]:
        heatmap = cv2.resize(heatmap, 
                           (original_image.shape[1], original_image.shape[0]))
    
    
    if heatmap.max() > 1.0:
        heatmap = heatmap / heatmap.max()
    
 
    heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]  
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
 
    overlay = (alpha * heatmap_colored + (1 - alpha) * original_image).astype(np.uint8)
    
    return overlay


def create_side_by_side_comparison(images: List[np.ndarray], 
                                   titles: List[str],
                                   save_path: Optional[str] = None) -> None:
   
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    
    if n == 1:
        axes = [axes]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

