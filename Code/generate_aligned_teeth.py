import numpy as np
import requests
import cv2
import replicate


def generate_aligned_mouth_replicate(teeth_contour: np.ndarray, mouth_mask: np.ndarray,
                                      face: np.ndarray) -> np.ndarray:
    """
    This function is used to run the GAN model on the input image and return the output image
    :param teeth_aligned_contour: Contours of aligned teeth
    :param mouth_mask: Mask of mouth region
    :param face: Image of face
    :return: generated mouth with aligned teeth
    """

    cv2.imwrite("teeth_contour.jpg", teeth_contour)
    cv2.imwrite("mouth_mask.jpg", mouth_mask)
    cv2.imwrite("face.jpg", face)

    teeth_contour_img = open(r"teeth_contour.jpg", "rb")
    mouth_mask_img = open(r"mouth_mask.jpg", "rb")
    face_img = open(r"face.jpg", "rb")

    image_payload = {
        "teeth_contour": teeth_contour_img,
        "mouth_mask": mouth_mask_img,
        "face": face_img
    }

    output = replicate.run(
        "techt3o/alignment_generation:07f913e02aa716977539d0e2c4e4c6a52e956ac5eb10411ebb7897a6ff0e0f4a",
        input=image_payload
    )
    img_data = requests.get(output).content
    arr = np.asarray(bytearray(img_data), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    return img