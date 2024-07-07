import yaml
import argparse
import os
import cv2
from Stage1_ToothSegm import Stage1
from Stage2_Mask2Mask import Stage2_Mask2Mask
from Stage3_Mask2Teeth import Stage3_Mask2Teeth
from Restore.Restore import Restore
import traceback
from matplotlib import pyplot as plt
import time

def main(image_path, args):
    start_time = time.time()
    img_name = os.path.basename(image_path).split('.')[0].split('/')[-1]
    visual = False
    stage_1_time = time.time()
    stage1_data = Stage1(image_path, mode=args.mode, state=args.stage1, if_visual=visual)
    print("Stage 1 time: ", time.time() - stage_1_time)
    stage_2_time = time.time()
    stage2_data = Stage2_Mask2Mask(stage1_data, mode=args.mode, state=args.stage2, if_visual=visual)
    print("Stage 2 time: ", time.time() - stage_2_time)
    stage2_data.update(stage1_data)
    stage_3_time = time.time()
    stage3_data = Stage3_Mask2Teeth(stage2_data, mode=args.mode, state=args.stage3, if_visual=visual)
    print("Stage 3 time: ", time.time() - stage_3_time)
    stage3_data.update(stage2_data)
    restore_time = time.time()
    pred = Restore(stage3_data['crop_mouth_align'], stage3_data)  # restore to original size
    print("Restore time: ", time.time() - restore_time)
    pred_face = pred['pred_ori_face']
    print("Total time: ", time.time() - start_time)

    ### save the visual results
    if not os.path.isdir(os.path.join(args.out_path, 'processing')):
        os.makedirs(os.path.join(args.out_path, 'processing'))
    if not os.path.isdir(os.path.join(args.out_path, 'prediction')):
        os.makedirs(os.path.join(args.out_path, 'prediction'))

    cv2.imwrite(os.path.join(os.path.join(args.out_path, 'prediction'), img_name + '.png'), pred_face)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.imread(image_path)[:, :, ::-1])
    plt.axis('off')
    plt.title('Input Image')
    plt.subplot(1, 2, 2)
    plt.imshow(pred_face[:, :, ::-1])
    plt.axis('off')
    plt.title('Predicted Image')
    # plt.tight_layout()
    plt.show()
    plt.savefig('output_' + img_name + '.png', bbox_inches='tight')

    for i, key in enumerate(
            ['crop_face', 'crop_mouth', 'crop_teeth', 'crop_teeth_align', 'cond_teeth_color', 'crop_mouth_align']):
        img = stage3_data[key]
        ### save together
        plt.subplot(3, 2, i+1)
        plt.imshow(img[:, :, ::-1])
        plt.axis('off')
    plt.savefig(os.path.join(os.path.join(args.out_path, 'processing'), img_name + '.png'), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--img_path', type=str, default='../Data/case1.jpg', help='path of the input facial photograph')
    with open("./Config.yaml", 'r') as f:
        GeneratorConfig = yaml.load(f, Loader=yaml.SafeLoader)['C2C2T_v2_facecolor_lightcolor']
    parser.set_defaults(**GeneratorConfig)
    args = parser.parse_args()
    folder_path = r"C:\Users\thpap\PycharmProjects\3D-Structure-guided-Network-for-Tooth-Alignment-in-2D-Photograph-modified\Data"
    # image_path = r"C:\Users\thpap\PycharmProjects\3D-Structure-guided-Network-for-Tooth-Alignment-in-2D-Photograph-modified\Data\t16.jpg"

    for image in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image)
        print(image_path)
        try:
            main(image_path, args)
        except Exception as e:
            print("An error occurred:", e)
            traceback.print_exc()
            continue


