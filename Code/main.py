import os
import cv2
from Stage1_ToothSegm import Stage1
from Stage2_Mask2Mask import Stage2_Mask2Mask
from Stage3_Mask2Teeth import Stage3_Mask2Teeth
from Restore.Restore import Restore
import traceback
from matplotlib import pyplot as plt
import time


def main(image_path):
    start_time = time.time()
    visual = False
    stage_1_time = time.time()
    stage1_data = Stage1(image_path, state="Stage1/SegmentToothContour/ckpt/ckpt_4800.pth", if_visual=visual)
    print("Stage 1 time: ", time.time() - stage_1_time)
    stage_2_time = time.time()
    teeth_contour, mouth_mask = stage1_data['crop_teeth'], stage1_data['crop_mask']
    aligned_teeth_contour = Stage2_Mask2Mask(teeth_contour, mouth_mask)
    print("Stage 2 time: ", time.time() - stage_2_time)
    stage_3_time = time.time()

    cv2.imwrite("teeth_align.jpg", aligned_teeth_contour)
    cv2.imwrite("mask.jpg", mouth_mask)
    cv2.imwrite("face.jpg", stage1_data['crop_face'])

    crop_mouth_align = Stage3_Mask2Teeth(aligned_teeth_contour, mouth_mask, stage1_data['crop_face'])
    print("Stage 3 time: ", time.time() - stage_3_time)
    restore_time = time.time()
    pred = Restore(crop_mouth_align, stage1_data)  # restore to original size
    print("Restore time: ", time.time() - restore_time)
    pred_face = pred['pred_ori_face']
    print("Total time: ", time.time() - start_time)
    return pred_face


if __name__ == '__main__':
    folder_path = r"C:\Users\thpap\PycharmProjects\3D-Structure-guided-Network-for-Tooth-Alignment-in-2D-Photograph-modified\Data"
    # image_path = r"C:\Users\thpap\PycharmProjects\3D-Structure-guided-Network-for-Tooth-Alignment-in-2D-Photograph-modified\Data\t16.jpg"

    for image in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image)
        print(image_path)
        try:
            pred_image = main(image_path)
            plt.imshow(pred_image[:, :, ::-1])
            plt.show()
        except Exception as e:
            print("An error occurred:", e)
            traceback.print_exc()
            continue


