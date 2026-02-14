import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

def find_unique_values(df):
    # ----- change labels -----
    unique_labels = df['change_type'].unique()

    # ----- construction status (across all dates) -----
    status_cols = [c for c in df.columns if c.startswith('change_status')]
    unique_status = pd.unique(df[status_cols].values.ravel())

    # ----- urban types (comma-separated, handle N,A) -----
    clean_urban = df['urban_type'].str.replace('N,A', 'NA_TEMP')
    unique_urban_types = (
        clean_urban
        .str.split(',')
        .explode()
        .str.strip()
        .unique()
    )
    unique_urban_types = [u.replace('NA_TEMP', 'N,A') for u in unique_urban_types]

    # ----- geography types (same logic) -----
    clean_geo = df['geography_type'].str.replace('N,A', 'NA_TEMP')
    unique_geo_types = (
        clean_geo
        .str.split(',')
        .explode()
        .str.strip()
        .unique()
    )
    unique_geo_types = [g.replace('NA_TEMP', 'N,A') for g in unique_geo_types]

    return (
        unique_labels,
        unique_urban_types,
        unique_geo_types,
        unique_status
    )


def preprocess_data(
    data,
    unique_labels,
    unique_urban_types,
    unique_geo_types,
    unique_status
):
    """
    Preprocesses the geospatial dataset into:
    X : feature matrix
    y : maps to integers
    """

    df = data.copy()

    # --------------------------------------------------
    # 1. Encode labels (Integer Mapping 0-5)
    # --------------------------------------------------
    label_map = {
        'Demolition': 0,
        'Road': 1,
        'Residential': 2,
        'Commercial': 3,
        'Industrial': 4,
        'Mega Projects': 5
    }

    # Use .map() to convert the strings to the specified integers
    # .fillna(-1) is a safety measure in case a label is missing
    y = df['change_type'].map(label_map).fillna(-1).astype(int).values

    # --------------------------------------------------
    # 2. Encode urban types (multi-label one-hot)
    # --------------------------------------------------
    urban_cols = []
    for u in unique_urban_types:
        col = f"urban_{u.replace(' ', '_').lower()}"
        df[col] = df['urban_type'].str.contains(u, regex=False).astype(int)
        urban_cols.append(col)

    # --------------------------------------------------
    # 3. Encode geography types (multi-label one-hot)
    # --------------------------------------------------
    geo_cols = []
    for g in unique_geo_types:
        col = f"geo_{g.replace(' ', '_').lower()}"
        df[col] = df['geography_type'].str.contains(g, regex=False).astype(int)
        geo_cols.append(col)

    # --------------------------------------------------
    # 4a. Encode construction status (per-date one-hot)
    # --------------------------------------------------
    status_cols = [c for c in df.columns if c.startswith('change_status')]

    status_feature_cols = []

    for t, status_col in enumerate(status_cols):
        for s in unique_status:
            col = f"status_t{t}_{str(s).replace(' ', '_').lower()}"
            df[col] = (df[status_col] == s).astype(int)
            status_feature_cols.append(col)

    # --------------------------------------------------
    # 4b. Time features: days between consecutive dates
    # --------------------------------------------------
    date_cols = [c for c in df.columns if c.startswith('date')]
    date_cols = sorted(date_cols)  # ensure chronological order

    dates = df[date_cols].apply(
        pd.to_datetime,
        errors='coerce',
        dayfirst=True
    )


    # Compute Δt between consecutive dates → 4 features
    delta_days = []
    for i in range(len(date_cols) - 1):
        delta = (dates.iloc[:, i + 1] - dates.iloc[:, i]).dt.days
        delta_days.append(delta.values.reshape(-1, 1))

    delta_days = np.hstack(delta_days)



    # --------------------------------------------------
    # 5. Scale image features
    # --------------------------------------------------
    img_cols = [c for c in df.columns if c.startswith('img_')]

    # --------------------------------------------------
    #6 Geometry feature: polygon area
    # --------------------------------------------------
   
    metric_geom = df.geometry.to_crs(epsg=6933)

    geom_features = np.column_stack([
        metric_geom.area.values,
        metric_geom.apply(lambda g: len(g.exterior.coords)).values,
        #metric_geom.length.values
    ])


    # --------------------------------------------------
    # 6. Concatenate all features
    # --------------------------------------------------
    X = np.hstack([
        df[img_cols].values, #30 floats
        geom_features, #2 floats
        delta_days, #4 floats
        df[urban_cols].values,  #12 binary
        df[geo_cols].values, #6 binary
        df[status_feature_cols].values, #11x5=55 binary
        
    ])
    #X should be (num_samples, num_features)=(296146 , 30+2+4+12+6+11x5=109)

    return X, y


def preprocess_test(
    data,
    unique_urban_types,
    unique_geo_types,
    unique_status
):
    """
    Preprocesses the geospatial dataset into:
    X : feature matrix

    """
    df = data.copy()
    
    # --------------------------------------------------
    # 2. Encode urban types (multi-label one-hot)
    # --------------------------------------------------
    urban_cols = []
    for u in unique_urban_types:
        col = f"urban_{u.replace(' ', '_').lower()}"
        df[col] = df['urban_type'].str.contains(u, regex=False).astype(int)
        urban_cols.append(col)

    # --------------------------------------------------
    # 3. Encode geography types (multi-label one-hot)
    # --------------------------------------------------
    geo_cols = []
    for g in unique_geo_types:
        col = f"geo_{g.replace(' ', '_').lower()}"
        df[col] = df['geography_type'].str.contains(g, regex=False).astype(int)
        geo_cols.append(col)

    # --------------------------------------------------
    # 4a. Encode construction status (per-date one-hot)
    # --------------------------------------------------
    status_cols = [c for c in df.columns if c.startswith('change_status')]

    status_feature_cols = []

    for t, status_col in enumerate(status_cols):
        for s in unique_status:
            col = f"status_t{t}_{str(s).replace(' ', '_').lower()}"
            df[col] = (df[status_col] == s).astype(int)
            status_feature_cols.append(col)

    # --------------------------------------------------
    # 4b. Time features: days between consecutive dates
    # --------------------------------------------------
    date_cols = [c for c in df.columns if c.startswith('date')]
    date_cols = sorted(date_cols)  # ensure chronological order

    dates = df[date_cols].apply(
        pd.to_datetime,
        errors='coerce',
        dayfirst=True
    )


    # Compute Δt between consecutive dates → 4 features
    delta_days = []
    for i in range(len(date_cols) - 1):
        delta = (dates.iloc[:, i + 1] - dates.iloc[:, i]).dt.days
        delta_days.append(delta.values.reshape(-1, 1))

    delta_days = np.hstack(delta_days)



    # --------------------------------------------------
    # 5. Scale image features
    # --------------------------------------------------
    img_cols = [c for c in df.columns if c.startswith('img_')]

    # --------------------------------------------------
    #6 Geometry feature: polygon area
    # --------------------------------------------------
   
    metric_geom = df.geometry.to_crs(epsg=6933)

    geom_features = np.column_stack([
        metric_geom.area.values,
        metric_geom.apply(lambda g: len(g.exterior.coords)).values,
        #metric_geom.length.values
    ])


    # --------------------------------------------------
    # 6. Concatenate all features
    # --------------------------------------------------
    X = np.hstack([
        df[img_cols].values, #30 floats
        geom_features, #2 floats
        delta_days, #4 floats
        df[urban_cols].values,  #12 binary
        df[geo_cols].values, #6 binary
        df[status_feature_cols].values, #11x5=55 binary
        
    ])
    #X should be (num_samples, num_features)=(296146 , 30+2+4+12+6+11x5=109)

    return X