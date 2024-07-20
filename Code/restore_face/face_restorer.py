import numpy as np
import cv2


def overlay_lips(original_img: np.ndarray, lips_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Overlay the lips on the original image with color matching and blending.
    :param original_img: Original image
    :param lips_img: Image to overlay on the lips
    :param mask: lip edge points
    :return: overlayed image
    """

    # Ensure both images are the same size
    lips_img = cv2.resize(lips_img, (original_img.shape[1], original_img.shape[0]))

    # Blur the mask to reduce hard edges
    blurred_mask = cv2.GaussianBlur(mask, (21, 21), 15)
    alpha = blurred_mask.astype(float) / 255  # Normalize to keep it in range [0, 1]

    # Enhance the prominence of the lips by increasing alpha in the lips area
    mask_strong = np.where(mask > 0, 1.3, 1.0)  # Increase opacity in the lips area
    alpha = np.clip(alpha * mask_strong, 0, 1)

    # Convert images to float for blending
    original_img_float = original_img.astype(float)
    lips_img_float = lips_img.astype(float)

    # Perform the blending
    blended_img = cv2.multiply(alpha, lips_img_float) + cv2.multiply(1 - alpha, original_img_float)
    blended_img = blended_img.astype(np.uint8)

    return blended_img


def get_mean_and_std(x):
    """
    Get mean and std of an image
    """
    x_mean, x_std = cv2.meanStdDev(x)
    x_mean = np.hstack(np.around(x_mean, 2))
    x_std = np.hstack(np.around(x_std, 2))
    return x_mean, x_std


def reinhard_color_transfer(input_img: np.ndarray, ref_img: np.ndarray):
    """Reinhards color transfer method"""
    s = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
    t = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB)
    s_mean, s_std = get_mean_and_std(s)
    t_mean, t_std = get_mean_and_std(t)

    height, width, channel = s.shape
    for i in range(0, height):
        for j in range(0, width):
            for k in range(0, channel):
                x = s[i, j, k]
                x = ((x-s_mean[k])*(t_std[k]/s_std[k]))+t_mean[k]
                # round or +0.5
                x = round(x)
                # boundary check
                x = 0 if x < 0 else x
                x = 255 if x > 255 else x
                s[i, j, k] = x

    s = cv2.cvtColor(s, cv2.COLOR_LAB2BGR)
    return s


def restore_image(mouth_align: np.ndarray, data: dict):
    """
    restore_face the image to the original size and overlay the aligned teeth on the face.
    """

    matched_img = reinhard_color_transfer(mouth_align.copy(), data['crop_mouth'].copy())

    # Blend the matched image with the original target image to smooth out artifacts
    alpha = 0.1  # Adjust alpha for blending, higher alpha means more of the original image is retained
    matched_img = cv2.addWeighted(data['crop_mouth'], alpha, matched_img, 1 - alpha, 0)

    ori_face = data['ori_face'].copy()
    detect_face = data['detect_face'].copy()
    crop_face = data['crop_face'].copy()
    crop_mask = data['crop_mask']
    info = data['info']

    info_0, info_1 = info[0], info[1]
    face_coord_x = info_0['coord_x']
    face_coord_y = info_0['coord_y']

    mouth_coord_x = info_1['coord_x']
    mouth_coord_y = info_1['coord_y']

    where = np.where((crop_mask[:, :, 0] == 255) & (crop_mask[:, :, 1] == 255) & (crop_mask[:, :, 2] == 255))

    crop_face[where[0], where[1]] = matched_img[where[0], where[1]]

    # detect_face with new teeth (numpy_BGR_uint8_512*512)
    smooth_overlaid_face = overlay_lips(detect_face[mouth_coord_y[0]:mouth_coord_y[1],
                                        mouth_coord_x[0]:mouth_coord_x[1], :].copy(),
                                        crop_face.copy(), crop_mask.copy())

    detect_face[mouth_coord_y[0]:mouth_coord_y[1], mouth_coord_x[0]:mouth_coord_x[1], :] = smooth_overlaid_face

    detect_face_resize = cv2.resize(detect_face, ori_face[face_coord_y[0]:face_coord_y[1],
                                                 face_coord_x[0]:face_coord_x[1], :].shape[:2][::-1])

    ori_face[face_coord_y[0]:face_coord_y[1], face_coord_x[0]:face_coord_x[1], :] = detect_face_resize

    return ori_face
