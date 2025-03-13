# -*- coding: utf-8 -*-
"""
Written by: Sina Jahangir
This code calculates the baseflow values based on HYSEP method(s)
Reference: HYSEP: A Computer Program for Streamflow Hydrograph Separation and Analysis
"""
import numpy as np
import pandas as pd


class BaseflowSeparation:
    def __init__(self, discharge_series, two_N_star):
        """
        Initialize the BaseflowSeparation class.
        
        :param discharge_series: Pandas Series with datetime index containing discharge values.
        :param two_N_star: Time interval parameter for HYSEP method.
        """
        self.discharge_series = discharge_series.to_numpy()
        self.index = discharge_series.index
        self.N = two_N_star
        self.baseflow_series = None
    
    def _find_local_minima(self):
        """
        Identify local minima in the discharge series using HYSEP local minimum method.
        """
        half_window = int(0.5 * (self.N - 1))
        minima_indices = []
        
        for i in range(len(self.discharge_series)):
            start = max(0, i - half_window)
            end = min(len(self.discharge_series), i + half_window + 1)
            window = self.discharge_series[start:end]
            if self.discharge_series[i] == np.min(window):
                minima_indices.append(i)
        
        return np.array(minima_indices)
    
    def _interpolate_baseflow_lm(self, minima_indices):
        """
        Interpolate baseflow between local minima using linear interpolation.
        """
        baseflow = np.interp(
            np.arange(len(self.discharge_series)),
            minima_indices,
            self.discharge_series[minima_indices]
        )
        return  np.minimum(baseflow, self.discharge_series)
    
    def separate_baseflow_lm(self):
        """
        Perform baseflow separation using the local minimum method.
        """
        minima_indices = self._find_local_minima()
        self.baseflow_series = self._interpolate_baseflow_lm(minima_indices)
        return self.baseflow_series
    
    def _sliding_interval_baseflow(self):
        """
        Apply the sliding-interval method to determine baseflow values.
        """
        half_window = int(0.5 * (self.N - 1))
        baseflow = np.zeros_like(self.discharge_series)
        
        for i in range(len(self.discharge_series)):
            start = max(0, i - half_window)
            end = min(len(self.discharge_series), i + half_window + 1)
            baseflow[i] = np.min(self.discharge_series[start:end])
        
        return  np.minimum(baseflow, self.discharge_series)
    
    def separate_baseflow_si(self):
        """
        Perform baseflow separation using the sliding-interval method.
        """
        return self._sliding_interval_baseflow()
    
    def _fixed_interval_baseflow(self):
        """
        Apply the fixed-interval method to determine baseflow values.
        """
        interval = self.N
        baseflow = np.zeros_like(self.discharge_series)
        
        for i in range(0, len(self.discharge_series), interval):
            end = min(i + interval, len(self.discharge_series))
            min_val = np.min(self.discharge_series[i:end])
            baseflow[i:end] = min_val
        
        return np.minimum(baseflow, self.discharge_series)
    
    def separate_baseflow_f(self):
        """
        Perform baseflow separation using the fixed-interval method.
        """
        return self._fixed_interval_baseflow()
#%%
#test examples
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    #sample CAMELS data
    df=pd.read_csv(r'D:\Paper\Code\EOF\AI-EOF\Sample data\CamelsRegionaltest_0_camels_01022500.csv')
    discharge_series=df.loc[:365,'q']
    # Apply baseflow separation
    N = 5
    
    
    baseflow_sep = BaseflowSeparation(discharge_series, N)
    baseflow_series = baseflow_sep.separate_baseflow_lm()
    
    # Plot results
    plt.figure(figsize=(5, 5),dpi=300)
    plt.plot(discharge_series.index, discharge_series, label="Total Discharge", color="blue")
    plt.plot(discharge_series.index, baseflow_series, label="Baseflow", color="red", linestyle="--")
    plt.xlabel("Sample")
    plt.ylabel("Discharge")
    plt.title("Baseflow Separation Using HYSEP Local Minimum Method")
    plt.legend()
    plt.show()
    
    baseflow_sep = BaseflowSeparation(discharge_series, N)
    baseflow_series_si = baseflow_sep.separate_baseflow_si()
    
    # Plot results
    plt.figure(figsize=(5, 5),dpi=300)
    plt.plot(discharge_series.index, discharge_series, label="Total Discharge", color="blue")
    plt.plot(discharge_series.index, baseflow_series_si, label="Baseflow", color="red", linestyle="--")
    plt.xlabel("Sample")
    plt.ylabel("Discharge")
    plt.title("Baseflow Separation Using HYSEP Sliding Interval Method")
    plt.legend()
    plt.show()
    baseflow_sep = BaseflowSeparation(discharge_series, N)
    baseflow_series_f = baseflow_sep.separate_baseflow_f()
    
    # Plot results
    plt.figure(figsize=(5, 5),dpi=300)
    plt.plot(discharge_series.index, discharge_series, label="Total Discharge", color="blue")
    plt.plot(discharge_series.index, baseflow_series_f, label="Baseflow", color="red", linestyle="--")
    plt.xlabel("Sample")
    plt.ylabel("Discharge")
    plt.title("Baseflow Separation Using HYSEP Fixed Interval Method")
    plt.legend()
    plt.show()
    
    
    
    
                
    
    
    

