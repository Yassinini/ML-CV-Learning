"""
Face Detection Using Haar Cascade
Consolidated from Haarcascade.ipynb
"""

import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path

# Get repo root for file paths
repo_root = Path(__file__).parent.parent
samples_dir = repo_root / 'samples' / 'Images'

# Load image
print("Loading image...")
img = cv.imread(str(samples_dir / 'people2.jpg'))
print("original shape, ", img.shape)
imgrey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print("gray shape, ", imgrey.shape)

# Face detection with Haar Cascade
print("\nRunning Haar Cascade face detection...")
facedetector = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
det = facedetector.detectMultiScale(imgrey, scaleFactor=1.2, minNeighbors=7)
print(f"Found {len(det)} faces")
print("Note: Inaccurate as there are actually 14 faces")


for (x, y, h, w) in det:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)


plt.figure(figsize=(10, 6))
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title("Haar Cascade Face Detection")
plt.axis("off")
plt.show()


img = cv.imread(str(samples_dir / 'people1.jpg'))
print("original image shape:", img.shape)
img = cv.resize(img, (1200, 1000))
print("resized image shape:", img.shape)
imgrey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


plt.figure(figsize=(10, 6))
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(imgrey, cmap='gray')
plt.title("Grayscale Image")
plt.axis("off")
plt.show()

# Eye and face detection with different parameters
print("\nRunning face and eye detection...")
eye_detector = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
fd = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')


dihtihtions = fd.detectMultiScale(imgrey, scaleFactor=1.25, minNeighbors=5)
print(f"First detection found {len(dihtihtions)} faces")

# Draw face rectangles
for (x, y, w, h) in dihtihtions:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Eye detection with specific parameters
idih = eye_detector.detectMultiScale(imgrey, scaleFactor=1.16, minNeighbors=8, maxSize=(44, 44))
print("Eye detection results:", idih)

# Second detection with different parameters
print("\nRefining detection parameters...")
for (x, y, w, h) in dihtihtions:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

idih = eye_detector.detectMultiScale(imgrey, scaleFactor=1.04   , minNeighbors=9, maxSize=(26, 26))
print("Refined eye detection:", idih)

# Draw eye rectangles
for (x, y, w, h) in idih:
    cv.rectangle(img, (x, y), (x + w, y + h), (140, 230, 10), 2)

# Display final results
plt.figure(figsize=(8, 6))
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Face and Eye Detection with Haar Cascade')
plt.axis("off")
plt.show()

