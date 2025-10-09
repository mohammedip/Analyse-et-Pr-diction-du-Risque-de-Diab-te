from data import df , pd
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.impute import KNNImputer
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



# traitement des valeur manquante  

df = df.replace(0, np.nan)  


imputer = KNNImputer(n_neighbors=5)  
df_filled = imputer.fit_transform(df)
new_df = pd.DataFrame(df_filled, columns=df.columns)

# traitement des valeurs aberrantes

new_df = new_df.drop(columns =["Unnamed: 0" ])

df_no_outliers = new_df.copy()


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


scaler = StandardScaler()
df_scaled = df_no_outliers.copy()
df_scaled = scaler.fit_transform(df_no_outliers)
df_scaled = pd.DataFrame(df_scaled, columns=df_no_outliers.columns, index=df_no_outliers.index)


for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    labels = kmeans.labels_
    score = silhouette_score(df_scaled, labels)
    print(f"K = {k}, Silhouette Score = {score}")



kmeans = KMeans(n_clusters=2, random_state=42)
df_scaled['Cluster'] = kmeans.fit_predict(df_scaled)    

cluster_means = df_scaled.groupby('Cluster').mean()
print(cluster_means)

cluster_count = df_scaled['Cluster'].value_counts()
print(cluster_count)

