
"""
Assignment B --> sustainability analysis in Python

By: Leshem Cohen

"""
#general packages
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
import seaborn as sns
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QSlider, QPushButton, QVBoxLayout, QMessageBox, QWidget
from PyQt5.QtCore import Qt
import matplotlib.patches as mpatches

#optimization packages
import sys
import cProfile, pstats
import time

#function imports
from assignment_B_functions import filter_regions
from assignment_B_functions import find_countries_with_fillable_gaps
from assignment_B_functions import univariate_imputation_mean
from assignment_B_functions import multivariate_imputation_uniform
from assignment_B_functions import multivariate_imputation_distance
from assignment_B_functions import calculate_pbias
from assignment_B_functions import normalized_rmse
from assignment_B_functions import r2
from assignment_B_functions import imputation_method_validation
from assignment_B_functions import stacked_bar_chart_1
from assignment_B_functions import stacked_bar_chart_2
from assignment_B_functions import show_threshold_selector

# primary indicator = methane emissions (kt of CO2 equivalent)
# secondary indicator = Agriculture land (sq km)

primary_indicator = pd.read_csv('C:/Users/leshe/OneDrive/Documents/IE/Year 2/SAPY/Assignment/Methane emissions/API_EN.ATM.METH.KT.CE_DS2_en_csv_v2_5872832.csv', header=2) 
secondary_indicator = pd.read_csv('C:/Users/leshe/OneDrive/Documents/IE/Year 2/SAPY/Assignment/Agriculture land sq km/API_AG.LND.AGRI.K2_DS2_en_csv_v2_5873155.csv', header=2)

#%% 4) Pre-process the data

#%% (a) Filtering data for a recent year

primary_indicator_filtered = primary_indicator.loc[:, ['Country Name', 'Country Code', '2020']]
secondary_indicator_filtered = secondary_indicator.loc[:, ['Country Name', 'Country Code', '2020']]

#change column name to make it easier for future analysis
primary_indicator_filtered = primary_indicator_filtered.rename(columns={'2020': '2020 primary'})
secondary_indicator_filtered = secondary_indicator_filtered.rename(columns={'2020': '2020 secondary'})

#%% (b) filtering data to remove regions and only keep countries

regions_list = ['AFE', 'AFW', 'ARB', 'CEB', 'CSS', 'EAP', 'EAR', 'EAS', 'ECA', 'ECS', 'EMU', 'EUU', 'FCS', 'HIC', 'HPC', 'IBD', 'IBT', 'IDA', 'IDB',
                     'IDX', 'INX', 'LAC', 'LCN', 'LDC', 'LIC', 'LMC', 'LMY', 'LTE', 'MEA', 'MIC', 'MNA', 'NAC', 'OED', 'OSS', 'PRE', 'PSS', 'PST', 
                     'SAS', 'SSA', 'SSF', 'SST', 'TEA', 'TEC', 'TLA', 'TMN', 'TSA', 'TSS', 'UMC', 'VEN', 'WLD']

primary_indicator_filtered, secondary_indicator_filtered = filter_regions(primary_indicator_filtered, secondary_indicator_filtered, regions_list)


#%% (c) Checking the number of mising values in both indicators

primary_indicator_missing_values = primary_indicator_filtered['2020 primary'].isna().sum()
secondary_indicator_missing_values = secondary_indicator_filtered['2020 secondary'].isna().sum()

print("total number of missing values in the primary indicator:")
print(primary_indicator_missing_values)

print("total number of missing values in the secondary indicator:")
print(secondary_indicator_missing_values)

#%% (d) determine the number of countries for which the data gaps in the primary indicator can be filled based on the secondary indicator

primary_col = '2020 primary'
secondary_col = '2020 secondary'

num_fillable, fillable_countries = find_countries_with_fillable_gaps(primary_indicator_filtered, secondary_indicator_filtered, primary_col, secondary_col)

print(f"Number of countries with fillable gaps: {num_fillable}")
print("Countries where primary indicator gaps can be filled by secondary indicator:")
print(fillable_countries)

#%% (e) Create a multi-index based on the two country-related attributes

primary_indicator_filtered.set_index(['Country Code', 'Country Name'], inplace=True)
secondary_indicator_filtered.set_index(['Country Code', 'Country Name'], inplace=True)

#%% 5) Perform the main analysis

#%% (a) fill missing values in the primary indicator based on univariate imputation with the average

primary_indicator_filled_univariate_mean = univariate_imputation_mean(primary_indicator_filtered, '2020 primary')

#%% (b) Multivariate imputation based on secondary indicator. Weighed uniformly to nearest neighbour

primary_indicator_filled_multivariate_uniform = multivariate_imputation_uniform(primary_indicator_filtered, secondary_indicator_filtered, '2020 primary', '2020 secondary', n_neighbors=5, weights='uniform')

#%% (c) Multivariate imputation based on secondary indicator. Weighed by distance to nearest neighour

primary_indicator_filled_multivariate_distance = multivariate_imputation_distance(primary_indicator_filtered, secondary_indicator_filtered, '2020 primary', '2020 secondary', n_neighbors=5, weights='distance')

#%% (d) validating the three different types of imputation with three criteria (R^2, NRMSE, and PBIAS) and 5-fold cross validation

# Drop NaNs from primary indicator and store the index of remaining rows
primary_indicator_filtered_obs = primary_indicator_filtered.dropna()
remaining_index_primary = primary_indicator_filtered_obs.index

# Filter secondary indicator based on the index of remaining rows in the primary indicator
secondary_indicator_filtered_obs = secondary_indicator_filtered.loc[remaining_index_primary]

validation_results = imputation_method_validation(primary_indicator_filtered_obs, secondary_indicator_filtered_obs)
    
#%% e) take the average of the performance metrics from the 5 rounds of cross-validation 

#Convert the tuple of lists into numpy arrays for each validation metric
r2_univariate = np.array(validation_results['R2_Univariate'])
nrmse_univariate = np.array(validation_results['NRMSE_Univariate'])
pbias_univariate = np.array(validation_results['PBIAS_Univariate'])

r2_multivariate_uniform = np.array(validation_results['R2_Multivariate_Uniform'])
nrmse_multivariate_uniform = np.array(validation_results['NRMSE_Multivariate_Uniform'])
pbias_multivariate_uniform = np.array(validation_results['PBIAS_Multivariate_Uniform'])

r2_multivariate_distance = np.array(validation_results['R2_Multivariate_Distance'])
nrmse_multivariate_distance = np.array(validation_results['NRMSE_Multivariate_Distance'])
pbias_multivariate_distance = np.array(validation_results['PBIAS_Multivariate_Distance'])

# Calculate the averages for each validation metric
mean_r2_univariate = np.mean(r2_univariate)
mean_nrmse_univariate = np.mean(nrmse_univariate)
mean_pbias_univariate = np.mean(pbias_univariate)

mean_r2_multivariate_uniform = np.mean(r2_multivariate_uniform)
mean_nrmse_multivariate_uniform = np.mean(nrmse_multivariate_uniform)
mean_pbias_multivariate_uniform = np.mean(pbias_multivariate_uniform)

mean_r2_multivariate_distance = np.mean(r2_multivariate_distance)
mean_nrmse_multivariate_distance = np.mean(nrmse_multivariate_distance)
mean_pbias_multivariate_distance = np.mean(pbias_multivariate_distance)
    
#%% f) collect the results from subtask e in a common data frame and print the results 

# Create a list to store the results for each imputation method
results = []

# Add the results for each imputation method
results.append(['Univariate (Mean)', mean_r2_univariate, mean_nrmse_univariate, mean_pbias_univariate])
results.append(['Multivariate (Uniform)', mean_r2_multivariate_uniform, mean_nrmse_multivariate_uniform, mean_pbias_multivariate_uniform])
results.append(['Multivariate (Distance)', mean_r2_multivariate_distance, mean_nrmse_multivariate_distance, mean_pbias_multivariate_distance])

# Create a DataFrame from the results list
results_df = pd.DataFrame(results, columns=['Imputation Method', 'Mean R2', 'Mean NRMSE', 'Mean PBIAS'])

# Print the results DataFrame
print("Results Summary:")
print(results_df)

#%% 6) Export the main results to a csv file, ensuring that it can be read nicely in a spreadsheet.

# Rename columns for better identification
primary_indicator_filtered_renamed = primary_indicator_filtered.rename(columns={'2020 primary': 'Methane emissions (kt CO2 equivalent)'})
primary_indicator_filled_univariate_mean_renamed = primary_indicator_filled_univariate_mean.rename(columns={'2020 primary': 'Imputed Univariate Mean'})
primary_indicator_filled_multivariate_uniform_renamed = primary_indicator_filled_multivariate_uniform.rename(columns={'2020 primary': 'Imputed Multivariate Uniform'})
primary_indicator_filled_multivariate_distance_renamed = primary_indicator_filled_multivariate_distance.rename(columns={'2020 primary': 'Imputed Multivariate Distance'})

# Combining data into a single DataFrame
combined_data = pd.concat([
    primary_indicator_filtered_renamed,
    primary_indicator_filled_univariate_mean_renamed,
    primary_indicator_filled_multivariate_uniform_renamed,
    primary_indicator_filled_multivariate_distance_renamed,
], axis=1)

# Resetting the index to include 'Country Code' and 'Country Name' as columns
combined_data.reset_index(inplace=True)

# Merging country-related information
combined_data = combined_data.merge(primary_indicator_filtered.reset_index(), on=['Country Code', 'Country Name'])
combined_data = combined_data.drop('2020 primary', axis=1)
# Exporting to a single CSV file
combined_data.to_csv('combined_data.csv', index=False)

#%% 7) make plots


#%% a) plot a histogram of the original primary indicator

plt.figure(figsize=(10, 6))
sns.histplot(primary_indicator_filtered['2020 primary'], bins=30,  kde=False, color='skyblue')
plt.xlabel('Methane emissions (kt of CO2 equivalent)')
plt.ylabel('Number of Countries')
plt.savefig('primary_indicator_histogram.png', dpi=150, bbox_inches='tight')
plt.close()

#%% a) plot a histogram of the original primary indicator (between interquartile ranges) for increased clarity on data distribution. 
# This will be used to effectively determine appropriate thresholds

first_quartile = np.percentile(primary_indicator_filtered_obs['2020 primary'], 10)
second_quartile = np.percentile(primary_indicator_filtered_obs['2020 primary'], 90)

plt.figure(figsize=(10, 6))
sns.histplot(primary_indicator_filtered['2020 primary'], bins=30, binrange=[235, 80249],  kde=False, color='skyblue')
plt.xlabel('Methane emissions (kt of CO2 equivalent)')
plt.ylabel('Number of Countries')
plt.savefig('primary_indicator_histogram_IQ_range.png', dpi=150, bbox_inches='tight')
plt.close()

#%% c) make a stacked bar chart of these segmentations that shows the proportions from 0 to 1

"""

the thresholds from this stacked bar chart cannot be changed from the GUI, they are the original 
thresholds observed from the histogram

"""

stacked_bar_chart_1(primary_indicator_filtered, primary_indicator_filled_univariate_mean,
                    primary_indicator_filled_multivariate_uniform, primary_indicator_filled_multivariate_distance)

#%% 8) GUI Element

"""
An interactive GUI element pops up here allowing you to choose high and low sustainability thresholds which
are subsequently used in the stacked bar chart. You can turn off the interactive mode by toggling betweein True
or False. 
"""

low_threshold, high_threshold = show_threshold_selector()


#%% 

"""
a stacked bar chart is created here based on the thresholds chosen from the GUI element. A low threshold of 
70 and a high threshold of 90 are the default threshold values based on the distribution of the histogram. 
"""

stacked_bar_chart_2(primary_indicator_filtered, primary_indicator_filled_univariate_mean,
          primary_indicator_filled_multivariate_uniform, primary_indicator_filled_multivariate_distance,
          low_threshold, high_threshold)


    




