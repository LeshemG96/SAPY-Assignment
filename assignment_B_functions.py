# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:40:05 2023

@author: leshe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QSlider, QPushButton, QVBoxLayout, QMessageBox, QWidget
from PyQt5.QtCore import Qt
import matplotlib.patches as mpatches


import seaborn as sns
import cProfile, pstats
import time
import sys


#%%

primary_indicator = pd.read_csv('C:/Users/leshe/OneDrive/Documents/IE/Year 2/SAPY/Assignment/Methane emissions/API_EN.ATM.METH.KT.CE_DS2_en_csv_v2_5872832.csv', header=2) 
secondary_indicator = pd.read_csv('C:/Users/leshe/OneDrive/Documents/IE/Year 2/SAPY/Assignment/Agriculture land sq km/API_AG.LND.AGRI.K2_DS2_en_csv_v2_5873155.csv', header=2)


#%% filtering data to remove regions and only keep countries
def filter_regions(primary_data, secondary_data, regions):
    
    """
    Filters out data from the primary and secondary datasets based on the provided regions.

    This function creates two conditions, one for each dataset, which check if the 'Country Code' 
    of each data point is not in the provided regions. It then filters the primary and secondary 
    datasets based on these conditions.

    Args:
        primary_data (pandas.DataFrame): The primary dataset to be filtered.
        secondary_data (pandas.DataFrame): The secondary dataset to be filtered.
        regions (list): A list of region codes to filter out from the datasets.

    Returns:
        tuple: A tuple containing two pandas.DataFrame objects. The first DataFrame is the filtered 
        primary dataset, and the second DataFrame is the filtered secondary dataset.

    Example:
        >>> primary_data = pd.DataFrame({'Country Code': ['US', 'CA', 'MX'], 'Value': [1, 2, 3]})
        >>> secondary_data = pd.DataFrame({'Country Code': ['US', 'CA', 'MX'], 'Value': [4, 5, 6]})
        >>> regions = ['US', 'CA']
        >>> filter_regions(primary_data, secondary_data, regions)
        ( Country Code Value
        2             MX    3
          Country Code Value
        2             MX    6,)
   """
    
    condition_1 = ~primary_data['Country Code'].isin(regions)
    condition_2 = ~secondary_data['Country Code'].isin(regions)

    primary_indicator_filtered = primary_data[condition_1]
    secondary_indicator_filtered = secondary_data[condition_2]

    return primary_indicator_filtered, secondary_indicator_filtered

#%% finding countries with fillable gaps

def find_countries_with_fillable_gaps(primary_data, secondary_data, primary_column, secondary_column):
      
    """
    Identifies countries with missing values in the primary indicator that can be filled with the secondary indicator.

     This function creates boolean masks for missing and non-missing values in the primary and secondary indicators,
     respectively. It then finds the countries where the primary indicator has missing values and the secondary 
     indicator has non-missing values. These countries can have their data gaps in the primary indicator filled 
     with the secondary indicator.

     Args:
         primary_data (pandas.DataFrame): The primary dataset.
         secondary_data (pandas.DataFrame): The secondary dataset.
         primary_column (str): The column in the primary dataset to check for missing values.
         secondary_column (str): The column in the secondary dataset to check for non-missing values.

     Returns:
         tuple: A tuple containing two elements. The first element is the number of countries for which data 
         gaps in the primary indicator can be filled based on the secondary indicator. The second element is a 
         pandas.DataFrame containing the countries that meet this condition.

     Example:
         >>> primary_data = pd.DataFrame({'Country': ['US', 'CA', 'MX'], 'Value': [1, np.nan, 3]})
         >>> secondary_data = pd.DataFrame({'Country': ['US', 'CA', 'MX'], 'Value': [4, 5, 6]})
         >>> find_countries_with_fillable_gaps(primary_data, secondary_data, 'Value', 'Value')
         (2, Country Value
          0    US  1.0
          1    CA  5.0)
     """
     
    # Create a boolean mask for missing values in the primary indicator
    primary_missing_values = primary_data[primary_column].isna()

    # Create a boolean mask for missing values in the secondary indicator
    secondary_missing_values = secondary_data[secondary_column].notna()

    # Find countries with missing values in the primary indicator that can be filled with the secondary indicator
    countries_with_fillable_gaps = secondary_data[primary_missing_values & secondary_missing_values]

    # Get the number of countries for which data gaps in the primary indicator can be filled based on the secondary indicator
    num_countries_with_fillable_gaps = len(countries_with_fillable_gaps)

    return num_countries_with_fillable_gaps, countries_with_fillable_gaps

#%% univariate imputation with mean

def univariate_imputation_mean(data, column_name):
    
    """
    
    Performs univariate imputation on a specific column of a DataFrame using the mean imputation strategy.

    This function uses the SimpleImputer from the sklearn.impute module with the strategy set to 'mean'. It makes a 
    copy of the input DataFrame, extracts the specified column, reshapes it for fitting, and then fits and transforms 
    the imputer on the extracted column. The imputed column replaces the original column in the copied DataFrame.

    Args:
        data (pandas.DataFrame): The DataFrame on which to perform the imputation.
        column_name (str): The name of the column on which to perform the imputation.

    Returns:
        pandas.DataFrame: A DataFrame where the specified column has been imputed using the mean imputation strategy.

    Example:
        >>> data = pd.DataFrame({'Value': [1, np.nan, 3]})
        >>> univariate_imputation_mean(data, 'Value')
        Value
        0  1.0
        1  2.0
        2  3.0
        
    """
       
    # Create a SimpleImputer with strategy 'mean'
    imputer = SimpleImputer(strategy='mean')

    # Make a copy of the DataFrame
    data_filled = data.copy()

    # Extract the column for imputation and reshape it for fitting
    column_to_impute = data_filled[column_name].values.reshape(-1, 1)

    # Fit and transform the imputer on the extracted column
    imputed_column = imputer.fit_transform(column_to_impute)

    # Replace the column in the copied DataFrame with the imputed values
    data_filled[column_name] = imputed_column

    return data_filled

#%% multivariate imputation (uniform)

def multivariate_imputation_uniform(primary_data, secondary_data, primary_column, secondary_column, n_neighbors=5, weights='uniform'):
    
    """
    Performs multivariate imputation on a specific column of a DataFrame using the KNN imputation strategy with uniform weighting.

    This function uses the KNNImputer from the sklearn.impute module with the specified number of neighbors and uniform weighting. It combines the specified columns from the primary and secondary datasets, fits and transforms the imputer on the combined data, and then assigns the imputed values back to the primary indicator.

    Args:
        primary_data (pandas.DataFrame): The primary dataset.
        secondary_data (pandas.DataFrame): The secondary dataset.
        primary_column (str): The column in the primary dataset to perform the imputation on.
        secondary_column (str): The column in the secondary dataset to perform the imputation on.
        n_neighbors (int, optional): The number of neighbors to use for the KNN imputation. Defaults to 5.
        weights (str, optional): The weight function used in prediction. Defaults to 'uniform'.

    Returns:
        pandas.DataFrame: A DataFrame where the specified column in the primary dataset has been imputed using the KNN imputation strategy with uniform weighting.

    Example:
        >>> primary_data = pd.DataFrame({'Value': [1, np.nan, 3]})
        >>> secondary_data = pd.DataFrame({'Value': [4, 5, 6]})
        >>> multivariate_imputation_uniform(primary_data, secondary_data, 'Value', 'Value')
        Value
        0 1.0
        1 5.0
        2 3.0
    """
   
    # Create a KNNImputer with the desired number of neighbors and uniform weighting
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)

    # Combine primary and secondary indicators
    combined_data = pd.concat([primary_data[primary_column], secondary_data[secondary_column]], axis=1)

    # Fit and transform the imputer on the combined data
    combined_data_filled = imputer.fit_transform(combined_data)

    # Assign the imputed values back to the primary indicator
    primary_data_imputed = primary_data.copy()
    primary_data_imputed[primary_column] = combined_data_filled[:, 0]

    return primary_data_imputed

#%% multivariate imputation (distance)

def multivariate_imputation_distance(primary_data, secondary_data, primary_column, secondary_column, n_neighbors=5, weights='distance'):
    
    """
    Performs multivariate imputation on a specific column of a DataFrame using the KNN imputation strategy with distance-based weighting.

    This function uses the KNNImputer from the sklearn.impute module with the specified number of neighbors and distance-based weighting. It combines the specified columns from the primary and secondary datasets, fits and transforms the imputer on the combined data, and then assigns the imputed values back to the primary indicator.

    Args:
        primary_data (pandas.DataFrame): The primary dataset.
        secondary_data (pandas.DataFrame): The secondary dataset.
        primary_column (str): The column in the primary dataset to perform the imputation on.
        secondary_column (str): The column in the secondary dataset to perform the imputation on.
        n_neighbors (int, optional): The number of neighbors to use for the KNN imputation. Defaults to 5.
     weights (str, optional): The weight function used in prediction. Defaults to 'distance'.

    Returns:
        pandas.DataFrame: A DataFrame where the specified column in the primary dataset has been imputed using the KNN imputation strategy with distance-based weighting.

    Example:
        >>> primary_data = pd.DataFrame({'Value': [1, np.nan, 3]})
        >>> secondary_data = pd.DataFrame({'Value': [4, 5, 6]})
        >>> multivariate_imputation_distance(primary_data, secondary_data, 'Value', 'Value')
        Value
        0 1.0
        1 5.0
        2 3.0
    """ 
    # Create a KNNImputer with the desired number of neighbors and distance-based weighting
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)

    # Combine primary and secondary indicators
    combined_data = pd.concat([primary_data[primary_column], secondary_data[secondary_column]], axis=1)

    # Fit and transform the imputer on the combined data
    combined_data_filled = imputer.fit_transform(combined_data)

    # Assign the imputed values back to the primary indicator
    primary_data_imputed = primary_data.copy()
    primary_data_imputed[primary_column] = combined_data_filled[:, 0]

    return primary_data_imputed

#%% validation calculations (PBIAS, R2, NRMSE)

# PBIAS function
def calculate_pbias(actual, predicted):
    return ((actual - predicted).sum() / actual.sum()) * 100

# NRMSE function
def normalized_rmse(observed, predicted):
  
    # Calculate RMSE
    rmse = np.sqrt(np.mean((predicted - observed) ** 2))

    # Calculate range of the observed values
    data_range = np.max(observed) - np.min(observed)

    # Calculate NRMSE
    nrmse = rmse / data_range

    return nrmse

# R2 function
def r2(observed, predicted):
  
    # Calculate the mean of observed values
    mean_observed = np.mean(observed)

    # Total sum of squares (TSS)
    total_sum_squares = np.sum((observed - mean_observed) ** 2)

    # Residual sum of squares (RSS)
    residual_sum_squares = np.sum((observed - predicted) ** 2)

    # Calculate R-squared
    r_squared = 1 - (residual_sum_squares / total_sum_squares)

    return r_squared



#%% validating the three different types of imputation with three criteria (R^2, NRMSE, and PBIAS) and 5-fold cross validation

def imputation_method_validation(primary_indicator_filtered_obs, secondary_indicator_filtered_obs):
    
    kf = KFold(n_splits=5, shuffle=True, random_state=False)

    r2_scores_univariate = []
    nrmse_scores_univariate = []
    pbias_scores_univariate = []

    r2_scores_multivariate_uniform = []
    nrmse_scores_multivariate_uniform = []
    pbias_scores_multivariate_uniform = []

    r2_scores_multivariate_distance = []
    nrmse_scores_multivariate_distance = []
    pbias_scores_multivariate_distance = []

    # Perform 5-fold cross-validation for all performance metrics
    for train_index, test_index in kf.split(primary_indicator_filtered_obs):
        primary_indicator_filtered_sim = primary_indicator_filtered_obs.copy()
        primary_indicator_filtered_sim.iloc[test_index, :] = np.nan
        secondary_indicator_filtered_sim = secondary_indicator_filtered_obs.copy()
        test_data = primary_indicator_filtered_obs.iloc[test_index, :].copy()

        #univariate imputation of simulated data
        primary_indicator_filtered_sim_univariate_mean = univariate_imputation_mean(primary_indicator_filtered_sim, '2020 primary')

        #multivariate imputation (uniform) of simulated data
        primary_indicator_filtered_sim_multivariate_uniform = multivariate_imputation_uniform(primary_indicator_filtered_sim, secondary_indicator_filtered_sim, '2020 primary', '2020 secondary', n_neighbors=5, weights='uniform')

        #multivariate imputation (distance) of simulated data
        primary_indicator_filtered_sim_multivariate_distance = multivariate_imputation_distance(primary_indicator_filtered_sim, secondary_indicator_filtered_sim, '2020 primary', '2020 secondary', n_neighbors=5, weights='distance')
        
        # Calculate performance metrics for univariate imputation (mean)
        r2_univariate = r2(test_data, primary_indicator_filtered_sim_univariate_mean.iloc[test_index, :])
        nrmse_univariate = normalized_rmse(test_data, primary_indicator_filtered_sim_univariate_mean.iloc[test_index, :])
        pbias_univariate = calculate_pbias(test_data, primary_indicator_filtered_sim_univariate_mean.iloc[test_index, :])
        
        # Calculate performance metrics for multivariate imputation (uniform)
        r2_multivariate_uniform = r2(test_data, primary_indicator_filtered_sim_multivariate_uniform.iloc[test_index, :]) 
        nrmse_multivariate_uniform = normalized_rmse(test_data, primary_indicator_filtered_sim_multivariate_uniform.iloc[test_index, :])
        pbias_multivariate_uniform = calculate_pbias(test_data, primary_indicator_filtered_sim_multivariate_uniform.iloc[test_index, :])
        
        #Calculate performance metrics for multivariate imputation (distance)
        r2_multivariate_distance = r2(test_data,  primary_indicator_filtered_sim_multivariate_distance.iloc[test_index, :])
        nrmse_multivariate_distance = normalized_rmse(test_data,  primary_indicator_filtered_sim_multivariate_distance.iloc[test_index, :])
        pbias_multivariate_distance = calculate_pbias(test_data,   primary_indicator_filtered_sim_multivariate_distance.iloc[test_index, :])
        
        # Append univariate performance metrics to appropriate list
        r2_scores_univariate.append(r2_univariate)
        nrmse_scores_univariate.append(nrmse_univariate)
        pbias_scores_univariate.append(pbias_univariate)
        
        # Append multivariate (uniform) performance metrics to appropriate list
        r2_scores_multivariate_uniform.append(r2_multivariate_uniform)
        nrmse_scores_multivariate_uniform.append(nrmse_multivariate_uniform)
        pbias_scores_multivariate_uniform.append(pbias_multivariate_uniform)
        
        # Append multivariate (distance) performance metrics to appropriate list
        r2_scores_multivariate_distance.append(r2_multivariate_distance)
        nrmse_scores_multivariate_distance.append(nrmse_multivariate_distance)
        pbias_scores_multivariate_distance.append(pbias_multivariate_distance)

    return {
        'R2_Univariate': r2_scores_univariate,
        'NRMSE_Univariate': nrmse_scores_univariate,
        'PBIAS_Univariate': pbias_scores_univariate,
        'R2_Multivariate_Uniform': r2_scores_multivariate_uniform,
        'NRMSE_Multivariate_Uniform': nrmse_scores_multivariate_uniform,
        'PBIAS_Multivariate_Uniform': pbias_scores_multivariate_uniform,
        'R2_Multivariate_Distance': r2_scores_multivariate_distance,
        'NRMSE_Multivariate_Distance': nrmse_scores_multivariate_distance,
        'PBIAS_Multivariate_Distance': pbias_scores_multivariate_distance
    }


#%% stacked bar chart with original thresholds observed from histogram

def stacked_bar_chart_1(primary_indicator_filtered, primary_indicator_filled_univariate_mean,
                        primary_indicator_filled_multivariate_uniform, primary_indicator_filled_multivariate_distance):
    
    low_threshold = primary_indicator_filtered['2020 primary'].quantile(0.70)
    high_threshold = primary_indicator_filtered['2020 primary'].quantile(0.90)

    def categorize(data):
        categories = pd.cut(data, bins=[-np.inf, low_threshold, high_threshold, np.inf], labels=['Low', 'Medium', 'High'])
        return categories

    primary_indicator_filtered['Category'] = categorize(primary_indicator_filtered['2020 primary'])
    primary_indicator_filled_univariate_mean['Category'] = categorize(primary_indicator_filled_univariate_mean['2020 primary'])
    primary_indicator_filled_multivariate_uniform['Category'] = categorize(primary_indicator_filled_multivariate_uniform['2020 primary'])
    primary_indicator_filled_multivariate_distance['Category'] = categorize(primary_indicator_filled_multivariate_distance['2020 primary'])

    methods = ['Original', 'Univariate', 'Multivariate Uniform', 'Multivariate Distance']
    data_frames = [primary_indicator_filtered, primary_indicator_filled_univariate_mean, primary_indicator_filled_multivariate_uniform, primary_indicator_filled_multivariate_distance]

    proportions_df = pd.DataFrame()
    
    colors = ['red', 'orange', 'blue']
    
    low_threshold_handle = mpatches.Patch(color=colors[0], label='Low Sustainability')
    medium_threshold_handle = mpatches.Patch(color=colors[1], label='Medium Sustainability')
    high_threshold_handle = mpatches.Patch(color=colors[2], label='High Sustainability')

    for method, df in zip(methods, data_frames):
        counts = df['Category'].value_counts(normalize=True)
        proportions_df[method] = counts

    proportions_df = proportions_df.reindex(['Low', 'Medium', 'High']).T

    proportions_df.plot(kind='bar', stacked=True, figsize=(10, 6), color=['red', 'orange', 'blue'])
    plt.ylabel('Proportion of Countries in Segment')
    plt.xlabel('Imputation Method')
    plt.xticks(rotation=0)
    plt.yticks(np.arange(0, 1.1, 0.1))
    #plt.legend(title='Methane Emissions (kt CO2 equivalent)', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(handles=[high_threshold_handle, medium_threshold_handle, low_threshold_handle], title='Thresholds', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('stacked_bar_chart_1_imputations.png', dpi=150, bbox_inches='tight')
    plt.close()


#%% creating a GUI element

def show_threshold_selector(interactive_mode=True):      
    class ThresholdSelector(QMainWindow):
        def __init__(self):
            super().__init__()
            # set window title and initialize the low and high thresholds
            self.setWindowTitle("Threshold Selector")
            self.low_threshold = 70
            self.high_threshold = 90
            
            #create sliders and window layout
            self.create_sliders()
            self.create_layout()
            
        def create_sliders(self):
            #low threshold slider creation 
            self.low_slider = QSlider(Qt.Horizontal, self)
            self.low_slider.setRange(0, 100)
            self.low_slider.setValue(self.low_threshold)
            self.low_slider.valueChanged.connect(self.update_low_label) #connecting the slider's value changed signal to the update_low_label method
            
            #high thredshold slider creation
            self.high_slider = QSlider(Qt.Horizontal, self)
            self.high_slider.setRange(0, 100)
            self.high_slider.setValue(self.high_threshold)
            self.high_slider.valueChanged.connect(self.update_high_label) #connecting the slider's value chandged signal to the update_high_label method

            #creating labels for the low and high thresholds
            self.low_label = QLabel(f"Low Threshold: {self.low_threshold}")
            self.high_label = QLabel(f"High Threshold: {self.high_threshold}")

            #description label
            self.description_label = QLabel("This tool allows you to set low and high thresholds of sustainability.")

        def create_layout(self):
            #create vertical layout
            layout = QVBoxLayout()
            layout.addWidget(self.description_label)
            layout.addWidget(self.low_label)
            layout.addWidget(self.low_slider)
            layout.addWidget(self.high_label)
            layout.addWidget(self.high_slider)
            
            #creating the ok button
            ok_button = QPushButton('OK')
            ok_button.clicked.connect(self.on_ok_button_clicked)
            layout.addWidget(ok_button)

            #create a central widget and set its layout to the created layout
            central_widget = QWidget()
            central_widget.setLayout(layout)
            self.setCentralWidget(central_widget)

        def update_low_label(self):
            self.low_threshold = self.low_slider.value()
            self.low_label.setText(f"Low Threshold: {self.low_threshold}")

        def update_high_label(self):
            self.high_threshold = self.high_slider.value()
            self.high_label.setText(f"High Threshold: {self.high_threshold}")

        def on_ok_button_clicked(self):
            # global low_threshold, high_threshold
            if self.low_slider.value() >= self.high_slider.value():
                QMessageBox.warning(self, 'Warning', 'Please select the thresholds in the correct order.')
            else:
                self.low_threshold = self.low_slider.value()
                self.high_threshold = self.high_slider.value()
                # low_threshold = self.low_threshold
                # high_threshold = self.high_threshold
                self.close()
                
    if interactive_mode:
        # global low_threshold, high_threshold
        app = QApplication(sys.argv)
        window = ThresholdSelector()
        window.show()
        app.exec_() # sys.exit(app.exec_())
        
    return window.low_threshold, window.high_threshold
        
#%% creating the stacked bar chart based on thresholds selected from the GUI

def stacked_bar_chart_2(primary_indicator_filtered, primary_indicator_filled_univariate_mean,
              primary_indicator_filled_multivariate_uniform, primary_indicator_filled_multivariate_distance,
              low_threshold, high_threshold):
    
    #calculating low and high quantiles based on the provided thresholds from the GUI
    low_quantile = primary_indicator_filtered['2020 primary'].quantile(low_threshold / 100)
    high_quantile = primary_indicator_filtered['2020 primary'].quantile(high_threshold / 100)
        
    #defining a function to categorize the data based on the calculated quantiles
    def categorize(data):
        categories = pd.cut(data, bins=[-np.inf, low_quantile, high_quantile, np.inf], labels=['Low', 'Medium', 'High'])
        return categories

    #applying the categorize function to each dataframe
    primary_indicator_filtered['Category'] = categorize(primary_indicator_filtered['2020 primary'])
    primary_indicator_filled_univariate_mean['Category'] = categorize(primary_indicator_filled_univariate_mean['2020 primary'])
    primary_indicator_filled_multivariate_uniform['Category'] = categorize(primary_indicator_filled_multivariate_uniform['2020 primary'])
    primary_indicator_filled_multivariate_distance['Category'] = categorize(primary_indicator_filled_multivariate_distance['2020 primary'])

    #provide the method and corresponding dataframes
    methods = ['Original', 'Univariate', 'Multivariate Uniform', 'Multivariate Distance']
    data_frames = [primary_indicator_filtered, primary_indicator_filled_univariate_mean, primary_indicator_filled_multivariate_uniform, primary_indicator_filled_multivariate_distance]

    proportions = pd.DataFrame()
    
    colors = ['red', 'orange', 'blue']
    
    #creating handles for the legend
    low_threshold_handle = mpatches.Patch(color=colors[0], label='Low sustainability')
    medium_threshold_handle = mpatches.Patch(color=colors[1], label='Medium sustainability')
    high_threshold_handle = mpatches.Patch(color=colors[2], label='High sustainability')

    #calculating the proportions for each method and category
    for method, df in zip(methods, data_frames):
        # print(df['Category'].head())
        counts = df['Category'].value_counts(normalize=True)
        proportions[method] = counts
        # print(counts)

    proportions = proportions.reindex(['Low', 'Medium', 'High']).T

    #plotting proportions as a stacked bar chart
    proportions.plot(kind='bar', stacked=True, figsize=(10, 6), color=['red', 'orange', 'blue'])
    plt.ylabel('Proportion of Countries in Segment')
    plt.xlabel('Imputation Method')
    plt.xticks(rotation=0)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend(handles=[high_threshold_handle, medium_threshold_handle, low_threshold_handle], title='Thresholds', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('stacked_bar_chart_2_imputations.png', dpi=150, bbox_inches='tight')
    plt.show()


        
        
