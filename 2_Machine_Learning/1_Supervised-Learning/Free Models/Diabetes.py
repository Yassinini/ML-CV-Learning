import kagglehub
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("mrsimple07/diabetes-prediction")
d=pd.read_csv(path + "/Diabetes_prediction.csv")

x=["Pregnancies", "Glucose", "BloodPressure", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
y="Diagnosis"
train_x , val_x, train_y , val_y = train_test_split(d[x], d[y], test_size=0.2)
m = LogisticRegression(max_iter=1000)
m.fit(train_x, train_y)

p=m.predict(val_x)
print(accuracy_score(val_y, p))

plt.figure(figsize=(10, 6))
plt.scatter(range(len(val_y)), p, alpha=0.5, color="blue", label="Predicted")
plt.scatter(range(len(val_y)), val_y, alpha=0.5, color='red', label='Actual')
plt.xlabel("Data Points")
plt.ylabel("Diagnosis")
plt.title("Diabetes Prediction")
plt.legend()
plt.show()