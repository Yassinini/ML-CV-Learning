import cv2 as cv
import dlib
import matplotlib.pyplot as plt
from pathlib import Path

# Get repo root for file paths
repo_root = Path(__file__).parent.parent
samples_dir = repo_root / 'samples'

car = cv.imread(str(samples_dir / 'Images' / 'car.jpg'))
car = cv.resize(car, (900, 900))
cagry = cv.cvtColor(car, cv.COLOR_BGR2GRAY)
cardih = cv.CascadeClassifier(str(samples_dir / 'Cascades' / 'cars.xml'))
card = cardih.detectMultiScale(cagry, scaleFactor=1.019, minNeighbors=6, minSize=(35, 35))

for (x, y, w, h) in card:
    cv.rectangle(car, (x, y), (x + w, y + h), (0, 0, 255), 2)

plt.figure(figsize=(10, 6))
plt.imshow(cv.cvtColor(car, cv.COLOR_BGR2RGB))
plt.title("Car Detection")
plt.axis('off')
#plt.show()


clock = cv.imread(str(samples_dir / 'Images' / 'clock.jpg'))
clock = cv.resize(clock, (800, 800))
cl = cv.cvtColor(clock, cv.COLOR_BGR2GRAY)
clockdetector = cv.CascadeClassifier(str(samples_dir / 'Cascades' / 'clocks.xml'))
clks = clockdetector.detectMultiScale(cl, scaleFactor=1.001, minSize=(85, 85), maxSize=(400, 400))

for (x, y, w, h) in clks:
    cv.rectangle(clock, (x, y), (x + w, y + h), (0, 0, 255), 2)

plt.figure(figsize=(10, 6))
plt.imshow(cv.cvtColor(clock, cv.COLOR_BGR2RGB))
plt.title("Clock Detection")
plt.axis('off')
#plt.show()

fbody = cv.imread(str(samples_dir / 'Images' / 'people3.jpg'))
fbody = cv.resize(fbody, (1703, 800))
detector = cv.CascadeClassifier(str(samples_dir / 'Cascades' / 'fullbody.xml'))
det = detector.detectMultiScale(fbody, scaleFactor=1.1, minSize=(110, 110))

for (x, y, w, h) in det:
    cv.rectangle(fbody, (x, y), (x + w, y + h), (0, 0, 255), 2)

plt.figure(figsize=(10, 6))
plt.imshow(cv.cvtColor(fbody, cv.COLOR_BGR2RGB))
plt.title("Full Body Detection")
plt.axis("off")
#plt.show()

#CNN && HOG & HAARCASCADE
mim=cv.imread(str(samples_dir / 'Images' / 'people1.jpg'))

cn=dlib.cnn_face_detection_model_v1(str(samples_dir / 'Weights' / 'mmod_human_face_detector.dat'))
cet=cn(mim, 1)

for face in cet:
    l,t,r,b,c= face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom(), face.confidence
    cv.rectangle(mim, (l,t), (r,b), (0,0,255),2)

plt.figure(figsize=(10, 6))
plt.imshow(cv.cvtColor(mim, cv.COLOR_BGR2RGB))
plt.title("People Detection CNN")
plt.axis('off')
plt.show()

#Haarcascade imported from test.ipynb file
img4=cv.imread(str(samples_dir / 'Images' / 'people1.jpg'))
hr=cv.CascadeClassifier(str(samples_dir / 'Cascades' / 'haarcascade_frontalface_default.xml'))
det = hr.detectMultiScale(img4, scaleFactor=1.1, minNeighbors=8)

for (x,y,w,h) in det:
    cv.rectangle(img4, (x, y), (x + w, y + h), (0, 0, 255), 2)

i=cv.CascadeClassifier(str(samples_dir / 'Cascades' / 'haarcascade_eye.xml'))
idet=i.detectMultiScale(img4, scaleFactor=1.1, minNeighbors=6, minSize=(23,23), maxSize=(75,75))
print(idet)
for (x,y,w,h) in idet:
    cv.rectangle(img4, (x, y), (x + w, y + h), (67, 200, 0), 2)

plt.figure(figsize=(10, 6))
plt.imshow(cv.cvtColor(img4, cv.COLOR_BGR2RGB))
plt.title("Face & Eye Detection HAARCASCADE")
plt.axis('off')
plt.show()

#HOG
heh=dlib.get_frontal_face_detector()
img4 = cv.imread(str(samples_dir / 'Images' / 'people1.jpg'))
deh=heh(img4, 4)

for face in deh:
    l,t,r,b=face.left(), face.top(), face.right(), face.bottom()
    cv.rectangle(img4, (l, t), (r,b), (0, 0, 255), 2)

plt.figure(figsize=(10, 6))
plt.imshow(cv.cvtColor(img4, cv.COLOR_BGR2RGB))
plt.title("Face Detection HOG")
plt.axis('off')
plt.show()