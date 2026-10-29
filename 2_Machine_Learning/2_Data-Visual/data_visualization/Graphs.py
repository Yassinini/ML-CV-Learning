from Plots import scatter, bar, line, filter_data, sort_data , countplot , heatmap , pairplot
#import kagglehub
#from kagglehub import KaggleDatasetAdapter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Get repo root for file paths
repo_root = Path(__file__).parent.parent.parent
data_dir = repo_root / 'Data Visual' / 'data' / 'data_visualization'


#path = kagglehub.dataset_download("anassarfraz13/housing-dataset-info-about-houses")
#housing = pd.read_csv(path + "/Housing.csv")
#scatter(housing, "area", "price")
#line(housing, "area", "price")


#path= kagglehub.dataset_download("dmahajanbe23/bmw-global-automotive-sales")
#bmw=pd.read_csv(path + "/bmw_global_sales_2018_2025.csv")
#scatter(bmw,"BEV_Share", "Units_Sold", True)

#path = kagglehub.dataset_download("zkskhurram/breast-cancer-stat-and-aware-dataset-2022-2025")
#Breast_cancer=pd.read_csv(path + "/breast_cancer_risk_factors.csv")
#line(Breast_cancer, "Risk_Factor", "Relative_Risk")

#Bcancer_Survival=pd.read_csv(path + "/breast_cancer_survival_by_stage.csv")
#line(Bcancer_Survival, "Stage", "One_Year_Survival_Pct")
#line(Bcancer_Survival, "Stage", "Five_Year_Survival_Pct")
#line(Bcancer_Survival, "Stage", "Ten_Year_Survival_Pct")
#line(Bcancer_Survival, "Stage", "Typical_Treatment")

#path = kagglehub.dataset_download("alexisbcook/data-for-datavis")
#Insurance=pd.read_csv(path + "/insurance.csv")
#scatter(Insurance, "age", "bmi", title="Age vs BMI, Insurance")

jordan_market_data = pd.read_csv(str(data_dir / 'jordan_market_dataset_2026.csv'))
#countplot(jordan_market_data, "Shoe_Model", "Condition")
#heatmap(jordan_market_data, "Resale_Price_USD", "coolwarm")
jordan_market_data.drop(columns=["Days_in_Inventory","Sale_Date"], inplace=True)
pairplot(jordan_market_data, "Profit_Margin_USD")

monthly_performance_data=pd.read_csv(str(data_dir / 'attachment_25168_session6_HW.csv.csv')) #hw from a self teaching course
plt.figure(figsize=(10,6))
plt.plot(monthly_performance_data["Sales"], monthly_performance_data["Month"])
plt.plot(monthly_performance_data["Expenses"], monthly_performance_data["Month"])
plt.title("Sales by Month")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,6))
plt.bar(monthly_performance_data["Customers"], monthly_performance_data["Month"])
plt.title("Customers by Month")
plt.xlabel("Month")
plt.ylabel("Customers")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,6))
plt.scatter(monthly_performance_data["Sales"], monthly_performance_data["Customers"])
plt.title("Sales by Customers")
plt.xlabel("Sales")
plt.ylabel("Customers")
plt.xticks(rotation=45)
plt.show()