"""
Utilities for emotion recognition
"""
import random
from typing import List, Tuple

import cv2
import numpy as np

IMAGE_WIDTH, IMAGE_HEIGHT = 416, 416
TARGET_SIZE = 48
COLOR_GREEN = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)


def class_colors(names):
    """
    Create a dict with one random BGR color for each
    class name
    """
    return {
        name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for name in names
    }


class Position:
    """
    Represents the position of an object
    """

    def __init__(self, left, top, width, height) -> None:
        self.left = left
        self.top = top
        self.width = width
        self.height = height

    @property
    def right(self):
        """Right position of the object, which is calculated as left + width"""
        return self.left + self.width

    @property
    def bottom(self):
        """Bottom position of the object, which is calculated as top + height"""
        return self.top + self.height

    def __str__(self) -> str:
        return f"Position(left={self.left}, top={self.top}, right={self.right}, bottom={self.bottom}, width={self.width}, height={self.height})"


class EmotionRecognition:
    """Class to recognise faces. Uses OpenCV under the hood to draw bounding boxes and load the model"""

    CLASSES = [
        "neutral",
        "happiness",
        "surprise",
        "sadness",
        "anger",
        "disgust",
        "fear",
        "contempt",
        "unknown",
    ]
    COLORS = class_colors(CLASSES)

    def __init__(self, face_recognition_model, emotion_recognition_model) -> None:
        self.face_recognition_model = face_recognition_model
        self.emotion_recognition_model = emotion_recognition_model

    def _refined_box(self, left, top, width, height):
        right = left + width
        bottom = top + height

        original_vert_height = bottom - top
        top = int(top + original_vert_height * 0.15)
        bottom = int(bottom - original_vert_height * 0.05)

        margin = ((bottom - top) - (right - left)) // 2
        left = (
            left - margin
            if (bottom - top - right + left) % 2 == 0
            else left - margin - 1
        )

        right = right + margin

        return left, top, right, bottom

    def draw_bounding_box(
        self,
        image: cv2.Mat,
        conf_threshold,
        pos: Position,
        label=None,
        with_emotion_label=True,
    ):
        """Draws a bounding box

        Args:
            image (cv2.Mat): Image to draw a bounding box on
            conf_threshold (_type_): Confidence threshold
            pos (Position): Position of the bounding box
        """
        cv2.rectangle(
            image, (pos.left, pos.top), (pos.right, pos.bottom), self.COLORS[label], 2
        )

        # text = f"{conf_threshold:.2f}"

        if with_emotion_label:
            # Display the label at the top of the bounding box
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            top = max(pos.top, label_size[1])
            cv2.putText(
                image,
                label,
                (pos.left, top - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                self.COLORS[label],
                1,
            )

    def predict_emotion(self, image):
        roi = cv2.resize(image, (TARGET_SIZE, TARGET_SIZE)) / 255
        predictions = self.emotion_recognition_model.predict(
            roi.reshape(-1, TARGET_SIZE, TARGET_SIZE, 3)
        )
        predicted_emotion = self.CLASSES[predictions.argmax(-1)[0]]
        return predicted_emotion

    def draw_bounding_boxes(self, detections, image, with_emotions=True):
        emotion = None
        for pos, confidence in detections:
            if with_emotions:
                roi = image[pos.top : pos.bottom, pos.left : pos.right]
                emotion = self.predict_emotion(roi)
            self.draw_bounding_box(image, confidence, pos, emotion)
        return image

    def detect_faces(
        self, image: cv2.Mat, conf_threshold: float, nms_threshold: float
    ) -> List[Tuple[Position, float]]:
        """Detect faces given an image

        Scan through all the bounding boxes output from the network and keep only
        those with high confidence scores. Assign the box's class label as the class
        with the highest score.

        Args:
            image (cv2.Mat): Image to detect
            conf_threshold (float): Confidence threshold
            nms_threshold (float): Non-max suppression threshold
        """
        blob = cv2.dnn.blobFromImage(
            image, 1 / 255, (IMAGE_WIDTH, IMAGE_HEIGHT), (0, 0, 0), True, crop=False
        )
        self.face_recognition_model.setInput(blob)

        # Predict faces
        outs = self.face_recognition_model.forward(
            self.face_recognition_model.getUnconnectedOutLayersNames()
        )
        frame_height = image.shape[0]
        frame_width = image.shape[1]

        confidences = []
        boxes: List[List[int, int, int, int]] = []
        recognitions: List[Tuple[Position, float]] = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant
        # overlapping boxes with lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        for i in indices:
            box = boxes[i]
            left, top, right, bottom = self._refined_box(box[0], box[1], box[2], box[3])
            bbox_pos = Position(
                max(left, 0), max(top, 0), max(right - left, 0), max(bottom - top, 0)
            )
            recognitions.append((bbox_pos, confidences[i]))
        return recognitions
