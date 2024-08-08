import joblib
import os
import numpy as np
import pandas as pd 
def load_model():
    path=r"C:\Users\ArtisusXiren\Desktop\Surprise_Housing\housing_predictor\myapp"
    xgb_model_path=os.path.join(path,'xgb_model.pkl')
    rf_model_path=os.path.join(path,'rf_model.pkl')
    mapping_path=os.path.join(path,'mappings.pkl')
    xgb_model=joblib.load(xgb_model_path)
    rf_model=joblib.load(rf_model_path)
    mappings=joblib.load(mapping_path)
    return xgb_model,rf_model,mappings
def pre_process(mappings,input_data):
    selected_features=['LotArea', 'LotShape', 'Neighborhood', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'Foundation', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Fireplaces', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageCond', 'WoodDeckSF', 'PoolArea']
    df=pd.DataFrame([input_data], columns=selected_features)
    for key, mapping in mappings.items():
      df_name,column=key.rsplit('_',1)
      if column in df.columns:
        df[column]=df[column].map(mapping)
    
    df=df[selected_features]
    return df  
def predict(model,df):  
    input_data=np.array(df.values).astype(float).reshape(1,-1)
    return model.predict(input_data)
    
    