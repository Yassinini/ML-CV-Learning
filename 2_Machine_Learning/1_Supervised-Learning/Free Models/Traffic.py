# Model works efficiently considering the data provided
import kagglehub
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("fedesoriano/traffic-prediction-dataset")
t=pd.read_csv(path + "/traffic.csv")

t["Datetime"]=pd.to_datetime(t["DateTime"])
t["hour"]=t["Datetime"].dt.hour
t["day"]=t["Datetime"].dt.day
t["month"]=t["Datetime"].dt.month
t["Vehicles"]=t["Vehicles"].fillna(15)
t = t[t["Vehicles"] < 115]

x=t[["hour", "day", "month", "Junction",]]
y=t["Vehicles"]
train_X , val_X , train_y , val_y = train_test_split(x,y, random_state=1, test_size=0.2)
model = RandomForestRegressor(n_estimators=120, random_state=1)
model.fit(train_X, train_y)
p=model.predict(val_X)

print(r2_score(val_y, p))

plt.figure(figsize=(10,6))
plt.scatter(val_X["hour"], val_y, label="Actual", color="red", alpha=0.2)
plt.scatter(val_X["hour"], p, label="Predicted", color="cyan", alpha=0.2)
plt.legend()
plt.xticks(rotation=45)
plt.show()