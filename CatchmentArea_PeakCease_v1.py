# -*- coding: utf-8 -*-
"""
Written by: Sina Jahangir
This code calculates the day numbers (N) where peak flow ceases based on an emprical formula.
The code also can be used to derive the filter size for baseflow seperation (2N*)
Peak flow refers to the maximum flow rate observed in a catchment
Reference: HYSEP: A Computer Program for Streamflow Hydrograph Separation and Analysis
"""
import numpy as np
class CatchmentArea:
    def __init__(self, area=1):
        """
        Parameters
        ----------
        area : float
            Catchment area in square km.

        Returns
        -------
        None.

        """
        self.area = area
    def report_cessation_days(self):
        """Calculate cessation days based on the area"""
        area = self.area
        cessation_days = 0.83 * area ** 0.2
        return cessation_days
    
    def closest_odd_integer(self, N):
        '''

        Parameters
        ----------
        N : int, float
            output of report_cessation_days.

        Returns
        -------
        int
            Closest int to 2*N.
        '''
        double_N_array = 2 * np.array(N)
        double_N_array = np.floor(double_N_array)
        #2N*
        closest_odd_array = np.where(double_N_array % 2 == 0, double_N_array + 1, double_N_array)
        return closest_odd_array
#%%
#test examples
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    catchment_area=np.arange(1,499)
    N=CatchmentArea(catchment_area).report_cessation_days()
    N_star=CatchmentArea().closest_odd_integer(N=N)
    plt.figure(dpi=300)
    plt.plot(catchment_area,N_star)
    plt.xlabel('Catchment Area')
    plt.ylabel('2N* (filter size)')
                
    
    
    

