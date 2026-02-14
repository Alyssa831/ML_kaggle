#%% import and load data
import geopandas as gpd
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from datapreprocess import find_unique_values, preprocess_data, preprocess_test
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

# %%
from sklearn.ensemble import RandomForestClassifier

# Use the FULL X (all 109 features), not the PCA version
rf = RandomForestClassifier(
    n_estimators=100,    # Number of trees
    max_depth=20,        # Limits tree height to prevent overfitting
    random_state=42,
    n_jobs=-1,           # Uses all your CPU cores for speed
    class_weight='balanced' # Fixes the 50% accuracy issue if classes are imbalanced
)

rf.fit(X, y) # X should be your full 109-feature matrix
#accuracy
p = rf.predict(X)
from sklearn.metrics import accuracy_score
Accuracy = accuracy_score(y, p) * 100.0
print(f"\n✅ RANDOM FOREST TRAINING ACCURACY ON FULL DATA: {Accuracy:.2f}%")

#%% use test data
df_test = gpd.read_file('test.geojson')

#%%
X_test = preprocess_test(
    df_test,
    unique_urban_types,
    unique_geo_types,
    unique_status
)
#%%

# Predict on test data
p_test = rf.predict(X_test)

# %%

# 3. Create the DataFrame for submission
# We use df_test.index as the 'Id' to match the sample format
submission = pd.DataFrame({
    'Id': range(len(p_test)), # Or use df_test['id'] if available
    'change_type': p_test
})

# 4. Save to CSV exactly as requested
submission.to_csv('sample_submission.csv', index=False)

print("✅ Submission file 'sample_submission.csv' created successfully!")
print(submission.head()) # Preview the first few rows
