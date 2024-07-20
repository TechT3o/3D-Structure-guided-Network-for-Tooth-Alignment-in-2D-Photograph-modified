import torch
import cv2
import numpy as np
from torchvision import transforms
from mouth_masking.SegmentToothContour.Model import UNet

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def segment_tooth_contour(mouth, state):
    """
    This function is used to segment the tooth contour from the mouth image on cpu
    """

    # Build model
    model = UNet(n_classes=2)
    model.load_state_dict(torch.load(state, map_location=torch.device('cpu')))
    model.to(torch.device('cpu'))
    model.eval()

    # Initialize data
    mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2RGB)    # numpy_RGB_uint8
    mouth = transform(mouth)
    mouth = mouth.unsqueeze(0).cpu()

    with torch.no_grad():
        pred = model(mouth)
        pred = pred[0].cpu().numpy().argmax(0)
        pred = np.uint8(pred*255)
        pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    return pred  #numpy_BGR_uint8

