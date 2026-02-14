#%% import and load data
import geopandas as gpd
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from datapreprocess import find_unique_values, preprocess_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pca import PCA

# Load the data
print("Loading train.geojson... (this might take a second)")
df_train = gpd.read_file('train.geojson')

#%% Example usage of preprocess_data function
unique_labels,unique_urban_types,unique_geo_types,unique_status=find_unique_values(df_train)
X, y = preprocess_data(
    df_train,
    unique_labels,
    unique_urban_types,
    unique_geo_types,
    unique_status
)
print(f"\n✅ PREPROCESSED DATA SHAPES: X={X.shape}, y={y.shape}")
# X should be (num_samples, num_features)=(296146 , 30+2+12+6+11x5+4=109)
# %%
print("\n--- SAMPLE PREPROCESSED FEATURES (FIRST) ---")
print(X[0])
print("\n--- SAMPLE PREPROCESSED LABELS (FIRST) ---")
print(y[0])

#%% Explore the dataset

# Draw the data in 3 dimensions (Hint: experiment with different combinations of dimensions)
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(X[:50,3],X[:50,1],X[:50,2], c=y[:50])
ax1.set_xlabel('1st dimension')
ax1.set_ylabel('2nd dimension')
ax1.set_zlabel('3rd dimension')
ax1.set_title("Vizualization of the dataset (3 dimensions)")
plt.show()

#%% PCA
import numpy as np
newData=[]
for k in [2,3,15,30]:
    #store newData for each k in a variable
    newdata = PCA(X[:,:30],k)
    newData.append(newdata)
    

#%% SVM
from SVM import SVM_LargeScale as SVM

for C in [0.1, 1.0, 10.0]:
    for i, newdata in enumerate(newData):
        Accuracy = SVM(newdata, y, C)
        print(f"\n✅ SVM TRAINING ACCURACY ON PCA-REDUCED DATA (k={[2,3,15,30][i]}): {Accuracy:.2f}%")
