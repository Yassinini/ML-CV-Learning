import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from pathlib import Path
from LBPH import lh as lbph_recognizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Get repo root for file paths
repo_root = Path(__file__).parent.parent.parent
samples_dir = repo_root / 'samples' / 'Datasets'

#Testing 1 Image
"""
test_image_path=str(samples_dir / 'yalefaces' / 'test' / 'subject07.happy.gif')
img=Image.open(test_image_path).convert('L')
image_array=np.array(img, 'uint8')

predicted_label=lbph_recognizer.predict(image_array)
expected_label=int(os.path.split(test_image_path)[1].split(".")[0].replace("subject", ""))
print("Predicted: ", predicted_label, "\n Expected: ", expected_label )

# Displaying the image
cv.putText(image_array, f"predict: , {predicted_label[0]}", (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
plt.figure(figsize=(6, 6))
plt.imshow(cv.cvtColor(image_array, cv.COLOR_BGR2RGB))
plt.title("Test Image")
plt.axis('off')
plt.show()
"""

# Testing Dataset
test_image_paths=[os.path.join(str(samples_dir / 'yalefaces' / 'test'), f) for f in os.listdir(str(samples_dir / 'yalefaces' / 'test'))]
predictions=[]
expected_labels=[]
for image_path in test_image_paths:
    img=Image.open(image_path).convert('L')
    input_image_array=np.array(img, 'uint8')
    predicted_label, _ =lbph_recognizer.predict(input_image_array)
    true_label= int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
    predictions.append(predicted_label)
    expected_labels.append(true_label)
predictions=np.array(predictions)
expected_labels=np.array(expected_labels)

print(f"Predicted: , {predictions} \n Expected:  {expected_labels}")
print(f"Accuracy: {accuracy_score(expected_labels, predictions)*100} %")
conf_matrix=confusion_matrix(expected_labels,predictions)

sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.show()