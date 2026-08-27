from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import kagglehub

path = kagglehub.dataset_download("anassarfraz13/housing-dataset-info-about-houses")
housing = pd.read_csv(path + "/Housing.csv")
x=housing[["area","bedrooms", "bathrooms","stories","parking"]]
y=housing.price 
m= RandomForestRegressor(random_state=1)
train_x , val_x , train_y, val_y=train_test_split(x,y,random_state=1)
m.fit(train_x, train_y)
pred=m.predict(val_x)
print(pred)
print(mean_absolute_error(val_y, pred))
print(housing["price"].mean())