# Note that Obesity is predictable when given the data needed
# High accuracy is due to the fact BMI is an easily predictable value
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import sys
import os
from pathlib import Path

# Add Data Visual module to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / 'Data Visual' / 'data_visualization'))
from Plots import scatter, line, bar
imp= SimpleImputer(strategy="mean")

path = kagglehub.dataset_download("mrsimple07/obesity-prediction")
a=pd.read_csv(path + "/obesity_data.csv")
a["ObesityCategory"] = a["ObesityCategory"].map({"Underweight" : -1, "Normal" : 0, "Overweight" : 1, "Obese" : 2})
a["Gender"]=a["Gender"].map({"Male" : 1, "Female" : 2})

x=a[["Weight", "Height", "Gender", "PhysicalActivityLevel", "Age"]]
x=imp.fit_transform(x)
y=a["ObesityCategory"].fillna(0)
x_train , x_val , y_train , y_val = train_test_split(x, y, test_size=0.4)
m=RandomForestClassifier(max_depth=5)
m.fit(x_train, y_train)
pred=m.predict(x_val)

p=m.predict([[75, 185, 1, 3,16]]) #Custom prediction 
print(p)

print("Accuracy Score:", accuracy_score(y_val, pred))
plt.figure(figsize=(10,6))
plt.scatter(x_val[:, 0], pred, alpha=0.2, label="Predicted")
plt.scatter(x_val[:, 0], y_val, alpha=0.2, label="Actual", color="red")
plt.xlabel("Weight")
plt.ylabel("Obesity Category")
plt.legend()
plt.title("Weight vs Obesity Category")
plt.show()