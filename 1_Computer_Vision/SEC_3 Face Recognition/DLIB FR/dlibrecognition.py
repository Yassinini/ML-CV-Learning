import cv2 as cv
import dlib
import matplotlib.pyplot as plt
from pathlib import Path

# Get repo root for file paths
repo_root = Path(__file__).parent.parent.parent
samples_dir = repo_root / 'samples'

face_detector=dlib.get_frontal_face_detector()
shape_predictor=dlib.shape_predictor(str(samples_dir / 'Weights' / 'shape_predictor_68_face_landmarks.dat'))
image=cv.imread(str(samples_dir / 'Images' / 'people2.jpg'))
detected_faces=face_detector(image,1)
for face in detected_faces:
    points=shape_predictor(image, face)
    for point in points.parts():
        cv.circle(image, (point.x, point.y), 2, (0,255,0), 1)
    print(points.parts())
    print(len(points.parts()))
    left , top , right , bottom = face.left(), face.top(), face.right(), face.bottom()
    cv.rectangle(image, (left,top), (right,bottom), (120,0,200), 2)
cv.imshow("dlib test", image)
cv.waitKey(0)
cv.destroyAllWindows()