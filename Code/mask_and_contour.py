from mouth_masking.DetectFace.DetectFace import detect_face
from mouth_masking.DetectMouth.DetectMouth import detect_mouth, crop_mouth
from mouth_masking.SegmentTeeth.DetectContour import masking_mouth
from mouth_masking.SegmentToothContour.SegmentToothContour import segment_tooth_contour
import numpy as np


def generate_mouth_mask_and_teeth_contour(img_path):
    """
    This function is used to generate mouth mask and teeth contour from the input image
    """
    ori_img, face, info_detectface = detect_face(img_path, newsize=(512, 512))
    face, mouth_mask, mouth_color = detect_mouth(face)
    crop_face, crop_mask, info_cropmouth = crop_mouth(face, mouth_mask, crop_size=(256, 128))

    mouth_masking = masking_mouth(crop_face, crop_mask)
    mouth_masking = np.uint8(mouth_masking)

    teeth_contour = segment_tooth_contour(mouth_masking, "mouth_masking/SegmentToothContour/ckpt/ckpt_4800.pth")
    teeth_contour = np.uint8(crop_mask/255 * teeth_contour)
    crop_teeth = teeth_contour

    return {
        "ori_face": ori_img,
        "detect_face": face,
        "info": {0: info_detectface, 1: info_cropmouth},
        "crop_face": crop_face,
        "crop_mouth": mouth_masking,
        "crop_teeth": crop_teeth,
        "crop_mask": crop_mask,
    }
