import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
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
    
    # Day of week feature (since day is numeric like 48, 49...)
    df_copy['day_of_week'] = df_copy['day'] % 7
    
    df_copy.drop('timestamp', axis=1, inplace=True)
    return df_copy

def preprocess_data(train_df, test_df):
    print("Preprocessing data with advanced features...")
    # Parse time and add cyclical features
    train_df = parse_time(train_df)
    test_df = parse_time(test_df)
    
    # Target transformation
    y_train = np.log1p(train_df['demand'])
    
    # Store indices for submission
    test_index = test_df['Index']
    
    # Drop target and index from training features
    X_train = train_df.drop(['demand', 'Index'], axis=1)
    X_test = test_df.drop(['Index'], axis=1)
    
    # Handle missing values
    # Temperature: fill with median
    temp_median = X_train['Temperature'].median()
    X_train['Temperature'] = X_train['Temperature'].fillna(temp_median)
    X_test['Temperature'] = X_test['Temperature'].fillna(temp_median)
    
    # Spatial features: Geohash substrings
    for X in [X_train, X_test]:
        X['geohash_5'] = X['geohash'].str[:5]
        X['geohash_4'] = X['geohash'].str[:4]
        X['geohash_3'] = X['geohash'].str[:3]
    
    # Categorical columns: fill with 'Missing'
    cat_cols = ['geohash', 'geohash_5', 'geohash_4', 'geohash_3', 'RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
    for col in cat_cols:
        X_train[col] = X_train[col].fillna('Missing').astype(str)
        X_test[col] = X_test[col].fillna('Missing').astype(str)
        
    return X_train, y_train, X_test, test_index, cat_cols

def train_and_predict():
    print("Loading datasets...")
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    
    X_train, y_train, X_test, test_index, cat_cols = preprocess_data(train_df, test_df)
    
    # Advanced Model parameters
    params = {
        'iterations': 3000,
        'learning_rate': 0.03,
        'depth': 9,
        'cat_features': cat_cols,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'verbose': 500,
        'early_stopping_rounds': 150
    }
    
    # K-Fold Cross Validation
    print("Starting K-Fold Cross Validation:")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))
    
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n--- Fold {fold + 1} ---")
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        model = CatBoostRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            use_best_model=True
        )
        
        val_preds = model.predict(X_va)
        oof_predictions[val_idx] = val_preds
        
        # Calculate R2 on original scale
        val_preds_orig = np.expm1(val_preds)
        y_va_orig = np.expm1(y_va)
        score = scaled_r2(y_va_orig, val_preds_orig)
        scores.append(score)
        print(f"Fold {fold + 1} Scaled R2 Score: {score:.4f}")
        
        # Add predictions for test set
        test_predictions += model.predict(X_test) / kf.n_splits
        
    avg_score = np.mean(scores)
    print(f"\nAverage Cross-Validation Scaled R2 Score: {avg_score:.4f}")
    
    # Final OOF score
    oof_orig = np.expm1(oof_predictions)
    y_train_orig = np.expm1(y_train)
    final_oof_score = scaled_r2(y_train_orig, oof_orig)
    print(f"Final Out-Of-Fold Scaled R2 Score: {final_oof_score:.4f}")
    
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
