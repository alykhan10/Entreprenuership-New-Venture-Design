import torch.nn as nn
import torchvision.models as models

def create_model(num_classes=4):
    # Load pre-trained MobileNetV2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # Modify the final classification layer
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model