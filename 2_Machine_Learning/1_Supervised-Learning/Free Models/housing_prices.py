from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add Data Visual module to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / 'Data Visual' / 'data_visualization'))
from Plots import line , scatter

path = kagglehub.dataset_download("anassarfraz13/housing-dataset-info-about-houses")
housing = pd.read_csv(path + "/Housing.csv")

x=housing[["area", "bedrooms", "bathrooms", "stories", "parking"]]
y=housing[["price"]]

m = LinearRegression()
m.fit(x,y)
predictions = m.predict(x)

print(r2_score(y,predictions))

plt.figure(figsize=(12,8))
plt.scatter(housing["area"], housing["price"], alpha=0.2, label="Actual")
plt.scatter(housing["area"], predictions, alpha=0.2, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Prices")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()


#t=m.predict([[1900, 3, 2, 2, 2]])
#print(m.predict([[1900, 3, 2, 2, 2]]))   #give data to evaluate
#print(r2_score(y, m.predict(x)))         #check accuracy

