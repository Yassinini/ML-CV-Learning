import cv2
import numpy as np

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
classifier_eye = cv2.CascadeClassifier("haarcascade_eye.xml")
camera = cv2.VideoCapture(0)
sample = 1
n_samples = 25
id = input('Type a number (ID): ')
width, height = 220, 220
print("Capturing the faces...")

while (True):
    conected, image = camera.read()
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(np.average(image_grey))
    detected_faces = classifier.detectMultiScale(image_grey,
                                                     scaleFactor=1.5,
                                                     minSize=(150,150))
    for (x, y, l, a) in detected_faces:
       cv2.rectangle(image, (x, y), (x + l, y + a), (0, 0, 255), 2)
       region = image[y:y + a, x:x + l]
       region_eye = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
       detected_eyes = classifier_eye.detectMultiScale(region_eye)
       for (ox, oy, ol, oa) in detected_eyes:
           cv2.rectangle(region, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)

           if cv2.waitKey(1) & 0xFF == ord('q'):
                if np.average(image_grey) > 110:
                    image_face = cv2.resize(image_grey[y:y + a, x:x + l], (width, height))
                    cv2.imwrite("person." + str(id) + "." + str(sample) + ".jpg", image_face)
                    print("[photo " + str(sample) + " captured successfully]")
                    sample += 1

    cv2.imshow("Face", image)
    cv2.waitKey(1)
    if (sample >= n_samples + 1):
        break

print("All faces captured successfully! ")
camera.release()
cv2.destroyAllWindows()