#%% import and load data
import geopandas as gpd
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler

# Load the data
print("Loading test.geojson... (this might take a second)")
df = gpd.read_file('test.geojson')

#%% Explore the dataset

#-------------------------------------------------------------------------
# 1. Verify Global Shape
#-------------------------------------------------------------------------

print(f"\n✅ DATASET SHAPE: {df.shape[0]} rows (polygons) x {df.shape[1]} columns")

#-------------------------------------------------------------------------
# 2. check column names
#-------------------------------------------------------------------------

print(f"\n✅ COLUMNS IN DATASET: {df.columns.tolist()}")

#-------------------------------------------------------------------------
# 3. Check the "Object" content (The Metadata)
#-------------------------------------------------------------------------

print("\n--- SAMPLE CONTENT (FIRST 5 ROWS) ---")
# This shows exactly what those strings look like
cols_to_see = ['urban_type', 'geography_type', 'change_type', 'geometry']
print(df[cols_to_see].head())
#find shape of geometry columns
print("\n--- COLUMN SHAPES ---")

#-------------------------------------------------------------------------
# 4. Understand the variety in your Labels
#-------------------------------------------------------------------------

print("\n--- UNIQUE CATEGORIES IN LABELS ---")
unique_labels = df['change_type'].unique()
#print shape and values
print(f"Total Unique Change Types: {len(unique_labels)}")
print(f"Unique Change Types: {unique_labels}")

# 1. Get the list of status columns
status_cols = [c for c in df.columns if c.startswith('change_status')]

# 2. Get unique values across ALL status columns at once
# We 'melt' them into one long series so we can find every possible label
unique_status = pd.unique(df[status_cols].values.ravel())

print("\n--- UNIQUE CATEGORIES IN Construction status ---")
print(f"Total Unique Change Status: {len(unique_status)}")
print(f"Unique Change Status: {unique_status}")

#-------------------------------------------------------------------------
# 5. Deep dive into the comma-separated strings
#-------------------------------------------------------------------------

#print("\n--- URBAN TYPE ---")
#print(df['urban_type'].unique())
#issuse! gives ['Residential, Commercial' 'Residential' 'N,A' 'Commercial'...

# Cleaning the 'urban_type' column to get unique values without spaces and handling 'N,A'
# Temporary fix: Replace 'N,A' with something without a comma, then split, then put it back
clean_series = df['urban_type'].str.replace('N,A', 'NA_TEMP')
unique_urban_types = clean_series.str.split(',').explode().str.strip().unique()
# Change 'NA_TEMP' back to 'N,A' in our final list
unique_urban_types = [t.replace('NA_TEMP', 'N,A') for t in unique_urban_types]
print("--- CLEANED UNIQUE URBAN TYPES ---")
#print shape and values
print(f"Total Unique Urban Types: {len(unique_urban_types)}")
print(unique_urban_types)

clean_series = df['geography_type'].str.replace('N,A', 'NA_TEMP')
unique_geo_types = clean_series.str.split(',').explode().str.strip().unique()
# Change 'NA_TEMP' back to 'N,A' in our final list
unique_geo_types = [t.replace('NA_TEMP', 'N,A') for t in unique_geo_types]
print("--- CLEANED UNIQUE GEOGRAPHY TYPES ---")
#print shape and values
print(f"Total Unique Geography Types: {len(unique_geo_types)}")
print(unique_geo_types)

#%%
#-------------------------------------------------------------------------
# 6. Inspect the Geometry details
#-------------------------------------------------------------------------

print("\n--- GEOMETRY INSPECTION ---")
# This shows the actual coordinate pairs for the first polygon
print(f"First Polygon Coordinates Sample:\n{df.geometry.iloc[5]}")
print(f"First Polygon Coordinates Sample:\n{df.geometry.iloc[7]}")

geom_df = df.geometry
# Project to a metric system (Unit = Meters)
metric_geom = df.geometry.to_crs(epsg=6933)
geom_features = np.column_stack([
    metric_geom.area.values,
    metric_geom.apply(lambda g: len(g.exterior.coords)).values,   
])
#see first 10
print(f"\nGeometry features sample (first 10 rows):\n{geom_features[:10]}")



# %%
