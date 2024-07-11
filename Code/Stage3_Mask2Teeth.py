import yaml
import torch
from Stage3.Network import Network
import numpy as np
from torchvision.utils import make_grid
import math


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.clamp_(*min_max)  # clamp
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = ((img_np+1) * 127.5).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type).squeeze()

def Stage3_Mask2Teeth(teeth_align, mask, face):

    from Stage3.Generator import Contour2ToothGenerator_FaceColor_LightColor as Generator
    with open("./Stage3/config/config_Contour2Tooth_facecolor_lightcolor.yaml", 'r') as f:
        GeneratorConfig = yaml.load(f, Loader=yaml.SafeLoader)['GeneratorConfig']
    # initialize the Network
    netG = Network(GeneratorConfig['unet'], GeneratorConfig['beta_schedule'])
    netG.load_state_dict(torch.load("Stage3/ckpt/ckpt_contour2tooth_v2_ContourSegm_facecolor_lightcolor_10000.pth"),
                         strict=False)
    netG.to(torch.device('cuda'))
    netG.eval()

    # initialize the Generator
    generator = Generator(netG)
    prediction = generator.predict(teeth_align, mask, face)       # tensor_BGR_float32 (-1to1)
    mouth_align = tensor2img(prediction)                          # numpy_BGR_uint8 (0-255)

    return mouth_align

