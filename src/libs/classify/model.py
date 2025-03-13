import torch.nn as nn
import torchvision.models as models

def create_model(num_classes=4, model_name="mobilenet_v2"):
    """
    Create a model based on the specified architecture.
    
    Args:
        num_classes: Number of output classes
        model_name: Model architecture to use
                    Options: "mobilenet_v2", "resnet50", "efficientnet_b0", "efficientnet_b3"
    
    Returns:
        torch.nn.Module: The model
    """
    
    if model_name == "mobilenet_v2":
        # Current model - MobileNetV2
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    elif model_name == "resnet50":
        # ResNet50 - significantly larger than MobileNetV2
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif model_name == "efficientnet_b0":
        # EfficientNet-B0 - better accuracy/parameter ratio
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    elif model_name == "efficientnet_b3":
        # EfficientNet-B3 - larger and more accurate than B0
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return model