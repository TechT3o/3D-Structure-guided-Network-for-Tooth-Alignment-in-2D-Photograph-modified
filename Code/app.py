from mask_and_contour import generate_mouth_mask_and_teeth_contour
from generate_aligned_teeth import generate_aligned_mouth_replicate
from restore_face.face_restorer import restore_image


def run_teeth_alignment(input_image):
    """
    This function is used to run the GAN model on the input image and return the output image
    """

    stage1_data = generate_mouth_mask_and_teeth_contour(input_image)
    crop_mouth_align = generate_aligned_mouth_replicate(stage1_data['crop_teeth'], stage1_data['crop_mask'],
                                                        stage1_data['crop_face'])
    aligned_teeth_face = restore_image(crop_mouth_align, stage1_data)  # restore to original size
    return aligned_teeth_face
