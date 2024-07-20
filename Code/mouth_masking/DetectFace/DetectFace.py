import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def face_landmark_detect(img, weight_path='./mouth_masking/DetectFace/ckpts/blaze_face_short_range.tflite'):
    """
    This function is used to detect face in the input image and return detection results
    """

    # Load detector
    base_options = python.BaseOptions(model_asset_path=weight_path)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    # Create mp Image object
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img.astype(np.uint8))

    # Detect face
    detection_result = detector.detect(image)
    print(detection_result.detections[0])
    if len(detection_result.detections) == 0:
        print("No faces detected.")
        raise ValueError("No faces detected.")

    if len(detection_result.detections) > 1:
        print("Multiple faces detected.")
        raise ValueError("Multiple faces detected.")

    if detection_result.detections[0].bounding_box.width <= 0 or detection_result.detections[0].bounding_box.height <= 0:
        raise ValueError("Not a valid face detection.")

    # Check the detection score (assuming the first category is the primary one)
    if detection_result.detections[0].categories[0].score < 0.4:
        raise ValueError("Detection score is too low.")

    for keypoint in detection_result.detections[0].keypoints:
        if not (0 <= keypoint.x <= 1) or not (0 <= keypoint.y <= 1):
            raise ValueError("Detection keypoints outside of bounds.")

    return detection_result.detections[0]


def detect_face(img, newsize=(512, 512)):
    """
    This function is used to detects the face in the input image and extract some face information from the results
    """
    detection = face_landmark_detect(img)

    bbox = detection.bounding_box
    x1, y1 = bbox.origin_x, bbox.origin_y
    x2, y2 = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

    face = img[y1:y2 + 1, x1:x2 + 1]
    face = cv2.resize(face, newsize)
    
    info = {
        'coord_x': (x1, x2+1),
        'coord_y': (y1, y2+1),
        'face_size': (x2+1-x1, y2+1-y1),
        'new_size': newsize,
    }

    return img, face, info