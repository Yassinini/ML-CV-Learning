import dlib
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path

# Get repo root for file paths
repo_root = Path(__file__).parent.parent
samples_dir = repo_root / 'samples'

img=cv.imread(str(samples_dir / 'Images' / 'people2.jpg'))

# HOG
fd=dlib.get_frontal_face_detector()
dt=fd(img,1)

for face in dt:
    l,t,r,b = face.left(), face.top(), face.right(), face.bottom()
    cv.rectangle(img, (l,t),(r,b), (0,0,255),2 )

plt.figure(figsize=(10,6))
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title("HOG Face Detection")
plt.axis("off")
plt.show()

# CNN
cnd=dlib.cnn_face_detection_model_v1(str(samples_dir / 'Weights' / 'mmod_human_face_detector.dat'))


det=cnd(img, 1)
for face in det:
    l,t,r,b,c = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom(), face.confidence
    cv.rectangle(img, (l,t), (r,b), (0,255,0), 2)

plt.figure(figsize=(10, 6))
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
plt.title("CNN Face Detection")
plt.axis('off')
plt.show()
