import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import pygeohash as pgh
import warnings

warnings.filterwarnings('ignore')

def scaled_r2(y_true, y_pred):
    return max(0, 100 * (r2_score(y_true, y_pred)))

def parse_time(df):
    df_copy = df.copy()
    time_split = df_copy['timestamp'].str.split(':', expand=True)
    df_copy['Hour'] = time_split[0].astype(int)
    df_copy['Minute'] = time_split[1].astype(int)
    df_copy['TimeInMinutes'] = df_copy['Hour'] * 60 + df_copy['Minute']
    
    # Cyclical time features
    df_copy['sin_TimeInMinutes'] = np.sin(2 * np.pi * df_copy['TimeInMinutes'] / 1440.0)
    df_copy['cos_TimeInMinutes'] = np.cos(2 * np.pi * df_copy['TimeInMinutes'] / 1440.0)
    
    # Day of week feature
    df_copy['day_of_week'] = df_copy['day'] % 7
    df_copy['is_weekend'] = (df_copy['day_of_week'] >= 5).astype(int)
    
    # Rush hour flag
    df_copy['is_rush_hour'] = ((df_copy['Hour'] >= 8) & (df_copy['Hour'] <= 10)) | ((df_copy['Hour'] >= 16) & (df_copy['Hour'] <= 19))
    df_copy['is_rush_hour'] = df_copy['is_rush_hour'].astype(int)
    
    df_copy.drop('timestamp', axis=1, inplace=True)
    return df_copy

def preprocess_data(train_df, test_df):
    print("Preprocessing data with advanced features and geohash decoding...")
    # Parse time and add cyclical features
    train_df = parse_time(train_df)
    test_df = parse_time(test_df)
    
    # Decode Geohashes to Latitude and Longitude
    def decode_geo(gh):
        try:
            return pgh.decode(gh)
        except:
            return (np.nan, np.nan)
            
    train_lat_lon = train_df['geohash'].apply(decode_geo).apply(pd.Series)
    train_df['latitude'] = train_lat_lon[0]
    train_df['longitude'] = train_lat_lon[1]
    
    test_lat_lon = test_df['geohash'].apply(decode_geo).apply(pd.Series)
    test_df['latitude'] = test_lat_lon[0]
    test_df['longitude'] = test_lat_lon[1]
    
    # Target transformation
    y_train = np.log1p(train_df['demand'])
    
    # Store indices for submission
    test_index = test_df['Index']
    
    # Drop target and index from training features
    X_train = train_df.drop(['demand', 'Index'], axis=1)
    X_test = test_df.drop(['Index'], axis=1)
    
    # Handle missing values
    # Temperature: fill with grouped median by day_of_week and Hour
    temp_grouped_median = X_train.groupby(['day_of_week', 'Hour'])['Temperature'].median()
    temp_global_median = X_train['Temperature'].median()

    def fill_temp(row):
        if pd.isna(row['Temperature']):
            try:
                return temp_grouped_median[row['day_of_week'], row['Hour']]
            except KeyError:
                return temp_global_median
        return row['Temperature']

    X_train['Temperature'] = X_train.apply(fill_temp, axis=1)
    X_test['Temperature'] = X_test.apply(fill_temp, axis=1)
    
    # Spatial features: Geohash substrings
    for X in [X_train, X_test]:
        X['geohash_5'] = X['geohash'].str[:5]
        X['geohash_4'] = X['geohash'].str[:4]
        X['geohash_3'] = X['geohash'].str[:3]
        X['geohash_2'] = X['geohash'].str[:2]
        X['geohash_1'] = X['geohash'].str[:1]
    
    # Categorical columns handling
    cat_cols = ['geohash', 'geohash_5', 'geohash_4', 'geohash_3', 'geohash_2', 'geohash_1', 'RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
    
    cb_X_train = X_train.copy()
    cb_X_test = X_test.copy()
    
    lgb_X_train = X_train.copy()
    lgb_X_test = X_test.copy()
    
    for col in cat_cols:
        # For CatBoost: fill 'Missing' and treat as strings
        cb_X_train[col] = cb_X_train[col].fillna('Missing').astype(str)
        cb_X_test[col] = cb_X_test[col].fillna('Missing').astype(str)
        
        # For LightGBM / XGBoost: fill 'Missing', label encode, and set to 'category'
        lgb_X_train[col] = lgb_X_train[col].fillna('Missing').astype(str)
        lgb_X_test[col] = lgb_X_test[col].fillna('Missing').astype(str)
        
        le = LabelEncoder()
        le.fit(list(lgb_X_train[col].unique()) + list(lgb_X_test[col].unique()))
        lgb_X_train[col] = le.transform(lgb_X_train[col])
        lgb_X_test[col] = le.transform(lgb_X_test[col])
        lgb_X_train[col] = lgb_X_train[col].astype('category')
        lgb_X_test[col] = lgb_X_test[col].astype('category')
        
    return y_train, test_index, cat_cols, cb_X_train, cb_X_test, lgb_X_train, lgb_X_test

def train_and_predict():
    print("Loading datasets...")
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    
    y_train, test_index, cat_cols, cb_X_train, cb_X_test, lgb_X_train, lgb_X_test = preprocess_data(train_df, test_df)
    
    # K-Fold Cross Validation
    print("Starting K-Fold Cross Validation with Blended Ensemble...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros(len(y_train))
    test_predictions = np.zeros(len(test_index))
    
    scores = []
    
    # Hyperparameters
    cb_params = {
        'iterations': 4000,
        'learning_rate': 0.05,
        'depth': 8,
        'l2_leaf_reg': 5,
        'bagging_temperature': 0.2,
        'cat_features': cat_cols,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'verbose': False,
        'early_stopping_rounds': 200
    }
    
    lgb_params = {
        'n_estimators': 4000,
        'learning_rate': 0.05,
        'max_depth': 8,
        'num_leaves': 63,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    xgb_params = {
        'n_estimators': 4000,
        'learning_rate': 0.05,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'hist',
        'enable_categorical': True,
        'early_stopping_rounds': 200,
        'n_jobs': -1
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(y_train)):
        print(f"\n--- Fold {fold + 1} ---")
        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # 1. CatBoost
        print("Training CatBoost...")
        cb_X_tr, cb_X_va = cb_X_train.iloc[train_idx], cb_X_train.iloc[val_idx]
        model_cb = CatBoostRegressor(**cb_params)
        model_cb.fit(cb_X_tr, y_tr, eval_set=(cb_X_va, y_va), use_best_model=True, verbose=False)
        cb_val_preds = model_cb.predict(cb_X_va)
        cb_test_preds = model_cb.predict(cb_X_test)
        
        # 2. LightGBM
        print("Training LightGBM...")
        lgb_X_tr, lgb_X_va = lgb_X_train.iloc[train_idx], lgb_X_train.iloc[val_idx]
        model_lgb = LGBMRegressor(**lgb_params)
        model_lgb.fit(lgb_X_tr, y_tr, eval_set=[(lgb_X_va, y_va)], callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)])
        lgb_val_preds = model_lgb.predict(lgb_X_va)
        lgb_test_preds = model_lgb.predict(lgb_X_test)
        
        # 3. XGBoost
        print("Training XGBoost...")
        model_xgb = XGBRegressor(**xgb_params)
        model_xgb.fit(lgb_X_tr, y_tr, eval_set=[(lgb_X_va, y_va)], verbose=False)
        xgb_val_preds = model_xgb.predict(lgb_X_va)
        xgb_test_preds = model_xgb.predict(lgb_X_test)
        
        # Blending (Average)
        val_preds = (cb_val_preds + lgb_val_preds + xgb_val_preds) / 3.0
        oof_predictions[val_idx] = val_preds
        
        val_preds_orig = np.expm1(val_preds)
        y_va_orig = np.expm1(y_va)
        score = scaled_r2(y_va_orig, val_preds_orig)
        scores.append(score)
        print(f"Fold {fold + 1} Ensemble Scaled R2 Score: {score:.4f}")
        
        test_predictions += (cb_test_preds + lgb_test_preds + xgb_test_preds) / 3.0 / kf.n_splits
        
    avg_score = np.mean(scores)
    print(f"\nAverage Cross-Validation Ensemble Scaled R2 Score: {avg_score:.4f}")
    
    # Final OOF score
    oof_orig = np.expm1(oof_predictions)
    y_train_orig = np.expm1(y_train)
    final_oof_score = scaled_r2(y_train_orig, oof_orig)
    print(f"Final Out-Of-Fold Ensemble Scaled R2 Score: {final_oof_score:.4f}")
    
    # Prepare submission
    print("\nPreparing submission...")
    final_test_predictions = np.expm1(test_predictions)
    
    submission = pd.DataFrame({
        'Index': test_index,
        'demand': final_test_predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    print(f"Submission saved to submission.csv. Shape: {submission.shape}")

if __name__ == "__main__":
    train_and_predict()
