import torch

def count_parameters(model):
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def extract_features(model, dataloader, device, num_classes):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y = torch.clamp(y, max=num_classes - 1)

            # Forward pass to get features from avgpool layer
            feat = model.module.relu(model.module.bn1(model.module.conv1(x)))
            feat = model.module.layer1(feat)
            feat = model.module.layer2(feat)
            feat = model.module.layer3(feat)
            feat = model.module.avgpool(feat)
            feat = feat.view(feat.size(0), -1)

            features.append(feat.cpu())
            labels.append(y.cpu())
    return torch.cat(features), torch.cat(labels)
