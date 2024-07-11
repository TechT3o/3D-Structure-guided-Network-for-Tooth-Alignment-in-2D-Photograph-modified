import torch
import numpy as np
import cv2
from torchvision import transforms


class Contour2ToothGenerator_FaceColor_LightColor():
    def __init__(self, network):
        super().__init__()
        self.netG = network.cuda()
        self.netG.set_new_noise_schedule()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.bf_image = data.get('bf_image').cuda()
        self.cond_image = data.get('cond_image').cuda()
        self.mask = data.get('mask').cuda()

    def Mask2TeethData_Process(self, teeth_contour, mask, face):
        teeth_contour = cv2.cvtColor(teeth_contour, cv2.COLOR_BGR2RGB)
        mask = np.array(mask)[:, :, 0] / 255
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        face_color = np.uint8(face * (1. - np.expand_dims(mask, -1)))

        teeth_contour = self.transform(teeth_contour)
        face_light_color_bar = self.transform(face_color)

        # construct the conditional image(2)
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        noisy_image = (1. - mask)*teeth_contour + mask*torch.randn_like(teeth_contour)
        cond_image = torch.cat([teeth_contour, face_light_color_bar, noisy_image], dim=0)


        out = {
            'bf_image': teeth_contour,  #three-channel 
            'cond_image': cond_image,   #seven-channel 
            'mask': mask,               #one-channel
            'cond_teeth_color': cv2.cvtColor(face_color, cv2.COLOR_RGB2BGR),
        }
        return out

    def predict(self, teeth_contour_align, mask, face):
        self.netG.eval()

        with torch.no_grad():

            out = self.Mask2TeethData_Process(teeth_contour_align, mask, face)
            bf_image = out['bf_image']
            cond_image = out['cond_image']
            mask = out['mask']

            self.set_input({
                'bf_image': bf_image.unsqueeze(0),
                'cond_image': cond_image.unsqueeze(0),   #four-channel 
                'mask': mask.unsqueeze(0),               #one-channel
            })

            self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=torch.randn_like(self.bf_image), y_0=self.bf_image, mask=self.mask, sample_num=1)
            prediction = torch.from_numpy(self.visuals[-1].detach().float().cpu().numpy()[::-1, ...].copy())     # torch_BGR_uint8
            return prediction
