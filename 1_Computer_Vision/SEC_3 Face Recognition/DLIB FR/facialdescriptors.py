import os
import cv2 as cv
import dlib
from PIL import Image
import numpy as np
from pathlib import Path

# Get the repo root (Machine_Learning folder) and construct paths
repo_root = Path(__file__).parent.parent.parent
weights_dir = repo_root / 'computer_vision' / 'Course_Materials' / 'Weights'
datasets_dir = repo_root / 'computer_vision' / 'Face Recognition' / 'samples' / 'Datasets' / 'yalefaces' / 'train'

face_detector=dlib.get_frontal_face_detector()
landmark_predictor=dlib.shape_predictor(str(weights_dir / 'shape_predictor_68_face_landmarks.dat'))
face_recognition_model=dlib.face_recognition_model_v1(str(weights_dir / 'dlib_face_recognition_resnet_model_v1.dat'))
descriptor_to_path_map={}
current_descriptor_index=0
face_descriptors=None
paths=[os.path.join(str(datasets_dir), f) for f in os.listdir(str(datasets_dir))]
for path in paths:
    img = Image.open(path).convert('RGB')
    image_array=np.array(img, 'uint8')
    detected_faces=face_detector(image_array,1)
    for face in detected_faces:
        l , t , r , b = face.left(), face.top(), face.right(), face.bottom()
        cv.rectangle(image_array, (l,t), (r,b), (100,0,200),2)
        points=landmark_predictor(image_array, face)
        for point in points.parts():
            cv.circle(image_array, (point.x, point.y), 2 , (0,255,0), 1)
        face_descriptor=face_recognition_model.compute_face_descriptor(image_array, points)
        face_descriptor=np.array(face_descriptor, np.float64)
        #print(face_descriptor)
        face_descriptor= face_descriptor[np.newaxis, :]
        #print(face_descriptor.shape) # (1,128)
        #print(face_descriptor) #matrices
        if face_descriptors is None:
            face_descriptors = face_descriptor
        else:
            face_descriptors = np.concatenate((face_descriptors, face_descriptor), axis =0)
        descriptor_to_path_map[current_descriptor_index]=path
        current_descriptor_index+=1
    print(face_descriptors.shape)
    cv.imshow("Detected Faces",image_array)
    cv.waitKey(0)
    cv.destroyAllWindows()