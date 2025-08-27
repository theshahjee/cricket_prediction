#model_utils.py

import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    """
    Feature engineering for cricket win prediction.
    Adds required runs, wickets remaining, required run rate, current run rate.
    """
    df = df.copy()
    
    # Ensure no negative balls_left
    df['balls_left'] = df['balls_left'].clip(lower=0)
    
    # Calculate features
    df['required_runs'] = df['target'] - df['total_runs']
    df['wickets_remaining'] = 10 - df['wickets']  # Assuming 10 wickets total
    
    # Calculate run rates
    overs_left = df['balls_left'] / 6
    overs_bowled = (120 - df['balls_left']) / 6
    overs_bowled = overs_bowled.replace(0, 1e-5)  # Avoid division by zero
    
    # Vectorized required run rate
    df['required_rr'] = np.where(
        df['balls_left'] > 0,
        df['required_runs'] / overs_left,
        np.where(df['required_runs'] > 0, np.inf, 0)
    )
    df['current_rr'] = df['total_runs'] / overs_bowled
    
    # Handle edge cases
    df['is_won_impossible'] = ((df['wickets'] >= 10) | 
                              (df['required_runs'] > df['balls_left'] * 1.5)).astype(int)
    
    return df

def get_features(df: pd.DataFrame) -> list:
    """Return list of feature columns."""
    return [
        'total_runs', 'wickets', 'target', 'balls_left',
        'required_runs', 'wickets_remaining', 'required_rr', 'current_rr', 'is_won_impossible'
    ]