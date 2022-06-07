import timm
import torch.nn as nn

def get_model(model_name, device, pretrained=True, num_classes=1000, freeze=False):
    model = timm.create_model(model_name, pretrained=pretrained, in_chans=1)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
            
    ##### set nn.Linear
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)
    model.to(device)
    
    return model