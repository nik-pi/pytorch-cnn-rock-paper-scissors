import torch
import torch.nn.functional as F

def predict_image(image, model, transform):
    image_tensor = transform(image).unsqueeze(0)
    with torch.inference_mode():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        _, predicted = torch.max(probabilities, 1)
    return predicted.item(), probabilities[0, predicted].item()
