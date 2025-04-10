import pandas as pd

def encode_column(df, column):
    """
    Encodes a categorical column using one-hot encoding.
    """
    return pd.get_dummies(df, columns=[column], drop_first=True)

def fill_missing_values(df):
    """
    Fills missing numeric values with the column's median.
    """
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    return df

def feature_scaling(df, method="minmax"):
    """
    Scales features using MinMaxScaler or StandardScaler.
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    
    scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled
