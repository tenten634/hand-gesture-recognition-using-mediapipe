#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import os
from collections import Counter
from collections import deque
import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.framework.formats import landmark_pb2

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier


def _open_camera(preferred_index: int, use_mac_backend: bool) -> cv.VideoCapture:
    # Prefer macOS AVFoundation backend first when explicitly requested
    if use_mac_backend:
        cap = cv.VideoCapture(preferred_index, cv.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            cap = cv.VideoCapture(preferred_index)
        if not cap.isOpened() and preferred_index != 0:
            cap.release()
            cap = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)
            if not cap.isOpened():
                cap.release()
                cap = cv.VideoCapture(0)
        return cap

    return cv.VideoCapture(preferred_index)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--mac', action='store_true',
                        help='Use macOS AVFoundation camera backend fallback')
    parser.add_argument("--hand_model_path", type=str,
                        default="model/hand_landmarker.task",
                        help="Path to MediaPipe hand landmarker model asset file")

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='Minimum confidence score for palm detection',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_presence_confidence",
                        help='Minimum confidence score for hand presence',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='Minimum confidence score for hand tracking',
                        type=float,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Argument parsing
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    use_mac_backend = args.mac
    hand_model_path = args.hand_model_path

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_presence_confidence = args.min_presence_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation
    cap = _open_camera(cap_device, use_mac_backend)
    if use_mac_backend and not cap.isOpened():
        print(f"Failed to open camera using macOS backend (device {cap_device}).")
        return
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model loading
    if not os.path.exists(hand_model_path):
        print("=" * 70)
        print("WARNING: Hand landmarker model asset not found!")
        print(f"Expected path: '{hand_model_path}'")
        print("")
        print("Please download the MediaPipe hand landmarker model from:")
        print("https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker")
        print("")
        print("Save the model file as 'hand_landmarker.task' in the 'model/' directory.")
        print("=" * 70)
        return

    # Create base options for the hand landmarker
    base_options = mp_tasks.BaseOptions(model_asset_path=hand_model_path)
    
    # Set running mode based on static_image_mode
    running_mode = (mp_vision.RunningMode.IMAGE if use_static_image_mode 
                    else mp_vision.RunningMode.VIDEO)
    
    # Create hand landmarker options
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=running_mode,
        num_hands=2,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    
    # Create hand landmarker
    hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Load labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS measurement module
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history
    finger_gesture_history = deque(maxlen=history_length)

    # Initialize mode and timestamp
    mode = 0
    frame_timestamp_ms = 0

    while True:
        fps = cvFpsCalc.get()

        # Key processing (ESC: exit)
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection execution
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        if use_static_image_mode:
            detection_result = hand_landmarker.detect(mp_image)
        else:
            # Timestamp in milliseconds for video mode
            detection_result = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            frame_timestamp_ms += int(1000 / 30)  # Assume ~30 FPS, increment by ~33ms

        # Process detection results
        if detection_result.hand_landmarks:
            for hand_landmarks, handedness in zip(detection_result.hand_landmarks,
                                                  detection_result.handedness):
                # Calculate bounding rectangle
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Calculate landmarks
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Convert to relative and normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Save training data
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Pointing gesture
                    point_history.append(landmark_list[8])  # Index finger coordinates
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculate most common gesture ID from recent detections
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Drawing
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, hand_landmarks)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Display update
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    # New API: landmarks is a list of NormalizedLandmark objects
    for landmark in landmarks:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoints
    # New API: landmarks is a list of NormalizedLandmark objects
    for landmark in landmarks:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to 1D list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalize
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to 1D list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmarks):
    """
    Draw hand landmarks using MediaPipe's built-in drawing utilities.
    
    Args:
        image: Image to draw on (BGR format)
        landmarks: List of normalized landmarks from MediaPipe HandLandmarker
    """
    if not landmarks:
        return image
    
    # Convert MediaPipe landmarks to protobuf format for drawing
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
        for landmark in landmarks
    ])
    
    # Use MediaPipe's drawing utilities
    mp.solutions.drawing_utils.draw_landmarks(
        image,
        hand_landmarks_proto,
        mp.solutions.hands.HAND_CONNECTIONS,
        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
        mp.solutions.drawing_styles.get_default_hand_connections_style()
    )
    
    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Bounding rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    # Extract handedness label from Category object
    # MediaPipe Tasks API: handedness is a list of Category objects
    # Get the first category (most confident one) and use category_name
    # Swap Left/Right because the image is flipped (mirrored) before MediaPipe processing
    try:
        if handedness and len(handedness) > 0:
            handedness_label = handedness[0].category_name
            # Swap Left and Right because the image is mirrored
            if handedness_label == 'Left':
                info_text = 'Right'
            elif handedness_label == 'Right':
                info_text = 'Left'
            else:
                info_text = handedness_label
        else:
            info_text = 'Unknown'
    except (AttributeError, TypeError, IndexError):
        info_text = 'Unknown'
    
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
