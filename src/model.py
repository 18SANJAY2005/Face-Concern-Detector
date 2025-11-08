import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List


class FaceConcernDetector(nn.Module):

    
    def __init__(self, num_concerns: int = 4, pretrained: bool = True, dropout_rate: float = 0.5):
     
        super(FaceConcernDetector, self).__init__()
        
        resnet = models.resnet18(pretrained=pretrained)
        
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        

        num_features = resnet.fc.in_features
        

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_concerns),
            nn.Sigmoid() 
        )
        
        self.num_concerns = num_concerns
        self.concern_names = ['acne', 'dark_circles', 'redness', 'wrinkles']
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        features = self.backbone(x)
        
        
        scores = self.classifier(features)
        
        return scores
    
    def predict(self, x: torch.Tensor) -> Dict[str, float]:
 
        self.eval()
        with torch.no_grad():
            scores = self.forward(x)
           
            scores_percent = (scores[0].cpu().numpy() * 100).astype(float)
            
            return {
                name: float(score) 
                for name, score in zip(self.concern_names, scores_percent)
            }


def load_model(model_path: str, device: str = 'cpu') -> FaceConcernDetector:
  
    model = FaceConcernDetector()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

