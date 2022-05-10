"""Main file to detect emotions"""
import argparse

import cv2
import numpy as np
import tensorflow as tf

from emotion_recognition import EmotionRecognition


def args_parser():
    """Arguments parser"""
    parser = argparse.ArgumentParser(description="YOLO emotion recognition")
    parser.add_argument(
        "--video_src",
        type=str,
        default=0,
        help="video source. If empty, uses webcam 0 stream",
    )
    parser.add_argument(
        "--image_input",
        type=str,
        default="",
        help="image source. Not used if empty",
    )
    parser.add_argument(
        "--out_image_filename",
        type=str,
        default="",
        help="output image file name. Not saved if empty",
    )
    parser.add_argument(
        "--out_video_filename",
        type=str,
        default="",
        help="inference video name. Not saved if empty",
    )
    parser.add_argument("--weights", default="yolov4.weights", help="yolo weights path")
    parser.add_argument(
        "--dont_show",
        action="store_true",
        help="windown inference display. For headless systems",
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=0.35,
        help="remove detections with confidence below this value",
    )
    parser.add_argument(
        "--nms_thresh",
        type=float,
        default=0.4,
        help="Non-max suppression threshold",
    )
    return parser.parse_args()


net = cv2.dnn.readNetFromDarknet(
    "cfg/yolov3-face.cfg", "weights/yolov3-wider_16000.weights"
)
TITLE = "Emotion recognition - COM3025"


def preprocess_image(img):
    """Preprocesses an image by converting to grayscale and expanding the dimensions from 1 to 3"""
    preprocessed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    preprocessed_img = np.stack((preprocessed_img,) * 3, axis=-1)
    return preprocessed_img


model = tf.keras.models.load_model("emotion_recognition")
emotion_recognition = EmotionRecognition(
    face_recognition_model=net, emotion_recognition_model=model
)

args = args_parser()

if not args.image_input:
    cap = cv2.VideoCapture(args.video_src)


if args.out_video_filename:
    video_writer = cv2.VideoWriter(
        args.out_video_filename,
        cv2.VideoWriter_fourcc(*"MJPG"),
        cap.get(cv2.CAP_PROP_FPS),
        (
            round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )


while True:
    if args.image_input:
        image = cv2.imread(args.image_input)
    else:
        ret, image = cap.read()
        if not ret:
            break
    image.flags.writeable = False
    gray_img = preprocess_image(image)
    faces_detected = emotion_recognition.detect_faces(
        gray_img, args.thresh, args.nms_thresh
    )
    emotion_recognition.draw_bounding_boxes(faces_detected, image, True)

    if not args.dont_show:
        cv2.imshow(TITLE, image)

    if args.out_image_filename:
        cv2.imwrite(args.out_image_filename, image)
    if args.out_video_filename:
        video_writer.write(image.astype(np.uint8))
    if cv2.waitKey(1) == ord("q"):
        break


# release resources
if not args.image_input:
    cap.release()
cv2.destroyAllWindows()
