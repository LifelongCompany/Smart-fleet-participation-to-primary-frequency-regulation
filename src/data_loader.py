import pandas as pd
import numpy as np
import os

def load_grid_data(filepath='data/france_2019_05.csv'):
    """
    Loads frequency data.
    Assumes data is Frequency Deviation in mHz.
    Returns Series in Hz (Deviation).
    """
    df = pd.read_csv(filepath, parse_dates=[0], index_col=0)
    # Convert mHz deviation to Hz deviation
    # The file has columns like ",Frequency" where Frequency is deviation in mHz
    df['Frequency'] = df['Frequency'] / 1000.0
    return df['Frequency']

def load_driving_data(filepath='data/driving_sessions.csv'):
    """
    Loads driving sessions.
    Parses timestamps to datetime.
    """
    df = pd.read_csv(filepath, sep=';')
    # Parse mixed formats if needed, or specify standard
    df['START'] = pd.to_datetime(df['START'], format='mixed', utc=True)
    df['STOP'] = pd.to_datetime(df['STOP'], format='mixed', utc=True)

    # Remove timezone info for simpler arithmetic with grid data (assuming grid is local/UTC consistent)
    df['START'] = df['START'].dt.tz_localize(None)
    df['STOP'] = df['STOP'].dt.tz_localize(None)

    return df

def load_obc_data(filepath='data/obc_efficiency.csv'):
    """
    Loads OBC efficiency curve.
    Returns DataFrame with [Power_kW, Efficiency].
    """
    # Skip first row if it's metadata, strictly parse headers
    # The previous code used skiprows=1, names=['Power_kW', 'Efficiency']
    df = pd.read_csv(filepath, sep=';', skiprows=1, names=['Power_kW', 'Efficiency'])
    df = df.sort_values('Power_kW').drop_duplicates('Power_kW')
    return df

def load_residual_value_data(filepath='data/residual_value.csv'):
    """
    Loads residual value data.
    """
    try:
        df = pd.read_csv(filepath, sep=';', encoding='latin1')
    except:
        df = pd.read_csv(filepath, sep=';', encoding='utf-8')
    return df
