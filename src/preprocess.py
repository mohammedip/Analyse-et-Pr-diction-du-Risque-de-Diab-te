from data import df , pd
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.impute import KNNImputer
import seaborn as sns


# traitement des valeur manquante  

df = df.replace(0, np.nan)  


imputer = KNNImputer(n_neighbors=5)  
df_filled = imputer.fit_transform(df)
new_df = pd.DataFrame(df_filled, columns=df.columns)

# traitement des valeurs aberrantes

new_df = new_df.drop(columns =["Unnamed: 0" ])

df_no_outliers = df.copy()


#Pregnancies
Q1 = df_no_outliers["Pregnancies"].quantile(0.25)
Q3 = df_no_outliers["Pregnancies"].quantile(0.75)
IQR = Q3 - Q1
Pregnancies_upper_band = Q3 + 2 * IQR
df_no_outliers = df_no_outliers[(df_no_outliers["Pregnancies"] <= Pregnancies_upper_band)]

#BloodPressure
Q1 = df_no_outliers["BloodPressure"].quantile(0.25)
Q3 = df_no_outliers["BloodPressure"].quantile(0.75)
IQR = Q3 - Q1
BloodPressure_lower_band = Q1 - 1.7 * IQR
BloodPressure_upper_band = Q3 + 1.7 * IQR

df_no_outliers.loc[df_no_outliers["BloodPressure"] > BloodPressure_upper_band, "BloodPressure"] = np.log1p(df_no_outliers.loc[df_no_outliers["BloodPressure"] > BloodPressure_upper_band, "BloodPressure"])

df_no_outliers = df_no_outliers[(df_no_outliers["BloodPressure"] >= BloodPressure_lower_band)]

#SkinThickness
Q1 = df_no_outliers["SkinThickness"].quantile(0.25)
Q3 = df_no_outliers["SkinThickness"].quantile(0.75)
IQR = Q3 - Q1
SkinThickness_upper_band = Q3 + 2.2 * IQR

df_no_outliers = df_no_outliers[(df_no_outliers["SkinThickness"] <= SkinThickness_upper_band)]


#Insulin
Q1 = df_no_outliers["Insulin"].quantile(0.25)
Q3 = df_no_outliers["Insulin"].quantile(0.75)
IQR = Q3 - Q1
Insulin_upper_band = Q3 + 4 * IQR

df_no_outliers = df_no_outliers[(df_no_outliers["Insulin"] <= Insulin_upper_band)]


#DiabetesPedigreeFunction
Q1 = df_no_outliers["DiabetesPedigreeFunction"].quantile(0.25)
Q3 = df_no_outliers["DiabetesPedigreeFunction"].quantile(0.75)
IQR = Q3 - Q1
DiabetesPedigreeFunction_upper_band = Q3 + 4 * IQR

df_no_outliers = df_no_outliers[(df_no_outliers["DiabetesPedigreeFunction"] <= DiabetesPedigreeFunction_upper_band)]


#Age
Q1 = df_no_outliers["Age"].quantile(0.25)
Q3 = df_no_outliers["Age"].quantile(0.75)
IQR = Q3 - Q1
Age_upper_band = Q3 + 2 * IQR

df_no_outliers = df_no_outliers[(df_no_outliers["Age"] <= Age_upper_band)]



