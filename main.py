import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import visualization

# settup
base_options = python.BaseOptions(model_asset_path='efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

#load camera
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if ret:
        detection_result = detector.detect(frame)
        frame_copy = np.copy(frame.numpy_view())
        annotated_image = visualization.visualize(frame_copy, detection_result)
        rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        cv2.imshow("detect", rgb_annotated_image)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break

video.release()
cv2.destroyAllWindows()
#v2
"""import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import visualization

# settup
base_options = python.BaseOptions(model_asset_path='efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

#load camera
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if ret:
        detection_result = detector.detect(frame, image_format=cv2.COLOR_BGR2RGB)
        annotated_image = visualization.visualize(frame, detection_result)
        cv2.imshow("detect", annotated_image)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break

"""