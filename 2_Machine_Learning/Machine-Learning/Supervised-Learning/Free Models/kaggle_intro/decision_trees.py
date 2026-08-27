from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import kagglehub

path = kagglehub.dataset_download("anassarfraz13/housing-dataset-info-about-houses")
housing = pd.read_csv(path + "/Housing.csv")

x = housing[["area","bedrooms","stories","parking"]]
y = housing["price"]
model = DecisionTreeRegressor(random_state=1)
train_X, val_X, train_Y, val_Y = train_test_split(x,y, random_state=2) #Train and validation split
model.fit(train_X, train_Y)

predict= model.predict(val_X) 
print(predict)
print(mean_absolute_error(val_Y, predict)) #model validation we look for mean absolute error