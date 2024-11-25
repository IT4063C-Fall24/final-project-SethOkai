#!/usr/bin/env python
# coding: utf-8

# # The problem of unemployed college graduates📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 Unemployment regarding college graduates

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 
# 📝 Are unemployment rates for college graduates rising or falling over time, and how do these rates compare to overall unemployment rates across different countries?

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 
# 📝 I hypothesize that college graduates’ unemployment varies tremendously from country to country depending on economic stability, labour market needs, and education. Economy and job markets in economies with strong employment such as Germany or Canada should be lower when compared with economies with weak economy or youth unemployment, such as Greece or Spain. In the US i think that college graduates’ unemployment rate is slightly lower than the overall unemployment rate but has been a bit more volatile due to economic downturns (global financial crisis of 2008, COVID-19, etc.) These spikes may be short-lived, but the longer-term picture could be a gradual drop in graduates’ unemployment as economies recover and meet new market demands, especially in skilled areas.

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝
# - URL(FILE): https://www.kaggle.com/datasets/pantanjali/unemployment-dataset
# - URL(FILE): https://fred.stlouisfed.org/series/CGBD2024
# - URL(API): https://www.bls.gov/developers/home.htm
# 
# In order to properly represent the issue of college graduates who don’t have jobs, the following variables from these datasets can be mapped:
# 
# **Unemployment Dataset**: This dataset has unemployment statistics for different demographic groups, such as education. Several variables of interest in integrating could be:
# - Year
# - State or Region
# - Degree Levels (e.g., Bachelor Degree)
# 
# **Federal Reserve Economic Data (FRED)**: This dataset tracks economic indicators like the labor market. The relevant variables to merge could be:
# - Year
# - State or Region
# - Rate Of Labour Force %
# 
# **BLS API**: The BLS API gives a wide range of labor market statistics, such as:
# - Year
# - Industry
# - Geographic Region (State, MSA)
# 
# ### Proposed Merging Strategy
# 
# Combining these datasets based on the related variables of Year and State or Region can give you insights into how college-graduate unemployment compares with other economic metrics and patterns. This collaboration will help further study the effect of education level on jobs and the way this works across regions/countries and time.

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 
# 📝I will use descriptive data to describe unemployment rates of college graduates,do trend analysis to look for trends across the years, between countries with more and less unemployment and Use visualizations like line graphs and bar graphs to show trends and country comparisons

# # Checkpoint 2

# ## Importing libraries and loading datasets

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import plotly.express as px


os.chdir('C:/Users/setho/OneDrive/Documents/GitHub/hello-github-okaiso-uc/final-project-SethOkai')

# Load the datasets
bls_data = pd.read_csv('extracted_files/bls_unemployment_data.csv') 
fred_data = pd.read_csv('extracted_files/fred_unemployment_data.csv')  
analysis_data = pd.read_csv('extracted_files/unemployment_analysis.csv')  


# ## Data cleaning
# ### Cleaned data by:
# - Handled missing or NaN values.
# - Renamed columns for clarity.
# - Converted 'Year' to datetime format.
# - Extracted year from 'Date' and create a new 'Year' column.
# - Dropped columns that may not be necessary.
# - Melted the DataFrame to have a long format for better analysis.
# - Observing if any data points doesnt fit the rest of the data: outliers

# In[2]:


# Loaded the datasets
bls_data = pd.read_csv('bls_unemployment_data.csv')
fred_data = pd.read_csv('fred_unemployment_data.csv')
analysis_data = pd.read_csv('unemployment_analysis.csv')

# Data Cleaning Function
def clean_bls_data(bls_df):
    # Check for missing values
    print("BLS Data Missing Values:\n", bls_df.isnull().sum())
    
    # Drop rows with missing 'Value' as it's crucial for analysis
    bls_df.dropna(subset=['Value'], inplace=True)
    
    # Convert 'Year' to datetime format
    bls_df['Year'] = pd.to_datetime(bls_df['Year'], format='%Y', errors='coerce')
    
    # Rename columns for clarity
    bls_df.rename(columns={'Value': 'BLS_Unemployment_Value'}, inplace=True)
    
    return bls_df

def clean_fred_data(fred_df):
    # Check for missing values
    print("FRED Data Missing Values:\n", fred_df.isnull().sum())
    
    # Drop rows with missing 'Unemployment Rate'
    fred_df.dropna(subset=['Unemployment Rate'], inplace=True)
    
    # Convert 'Date' to datetime format
    fred_df['Date'] = pd.to_datetime(fred_df['Date'], errors='coerce')
    
    # Extract year from 'Date' and create a new 'Year' column
    fred_df['Year'] = fred_df['Date'].dt.year
    
    return fred_df

def clean_analysis_data(analysis_df):
    # Check for missing values
    print("Analysis Data Missing Values:\n", analysis_df.isnull().sum())
    
    # Drop columns that may not be necessary 
    analysis_df.drop(columns=['Country Code'], inplace=True)
    
    # Melt the DataFrame to have a long format for better analysis
    analysis_df = analysis_df.melt(id_vars=['Country Name'], var_name='Year', value_name='Unemployment Rate')
    
    # Convert 'Year' to integer
    analysis_df['Year'] = analysis_df['Year'].astype(int)
    
    # Drop rows with missing 'Unemployment Rate'
    analysis_df.dropna(subset=['Unemployment Rate'], inplace=True)
    
    return analysis_df

# Clean the datasets
bls_data_cleaned = clean_bls_data(bls_data)
fred_data_cleaned = clean_fred_data(fred_data)
analysis_data_cleaned = clean_analysis_data(analysis_data)


# Display cleaned data
print("\nCleaned BLS Data:\n", bls_data_cleaned.head())
print("\nCleaned FRED Data:\n", fred_data_cleaned.head())
print("\nCleaned Analysis Data:\n", analysis_data_cleaned.head())


# ## statistical summaries of the data.
# In the BLS dataset 177 values of unemployment are given for a period. The unemployment rate is about 3.06%, which is pretty low in the years that are known. The data is from a low of 1.80% up to a high of 8.40% which indicates it varies over time. On the 25th percentile, 25 per cent of the data has rates under 2.20%, median is 2.50%, meaning half of the unemployment rates reported are below that figure. And the 75th percentile indicates that 75% of data are less than 3.90% which means the data distribution is at the upper end. There is a standard deviation (same as 1.14%) between the unemployment numbers on the average.
# 
# The FRED dataset by contrast contains 57 records for unemployment rates and year information. The average unemployment rate is 4.99% here which indicates low unemployment during the sample period. This record has the lowest rate at 3.40% and the highest at 14.80%, which shows there are big shifts in the labor market over this period. The 25th percentile represents the 25 % unemployment rate which is lower than 3.60% and the median is 3.90%, which shows most of the data points are below 3.60%. And even the 75th percentile (5.580%) illustrates how unemployment increases in later data years. 2.41% Standard deviation: There is more variation in unemployment rates than in the BLS data. These statistics provide an overview of unemployment, that is a snapshot of the economy over time, and it is used as a starting point for further research or simulation.

# In[ ]:


print("BLS Data Descriptive Statistics:")
print(bls_data_cleaned.describe())
print("\n" + "-"*50 + "\n")

print("FRED Data Descriptive Statistics:")
print(fred_data_cleaned.describe())
print("\n" + "-"*50 + "\n")

print("Analysis Data Descriptive Statistics:")
print(analysis_data_cleaned.describe())


# ##  Line Chart for BLS Unemployment Data
# **Description: This plot shows decades-based unemployment data from the Bureau of Labor Statistics (BLS). It indicates the way unemployment has changed from year to year.**
# 
# **Summary: This graph shows some key indicators of U.S. unemployment from years past. Deep spikes could be times of financial crisis, and steep dips could be times of economic recovery.**

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
bls_data_cleaned['Year'] = bls_data_cleaned['Year'].astype(str)
plt.figure(figsize=(12, 6))
sns.lineplot(data=bls_data_cleaned, x='Year', y='BLS_Unemployment_Value')
plt.title('BLS Unemployment Data Over Years')
plt.xlabel('Year')
plt.ylabel('Unemployment Value')
plt.xticks(rotation=45)
plt.grid()
plt.show()



# ## Line Chart for FRED Unemployment Data
# **Description: This plot shows decades-based unemployment data from the Federal Reserve Economic Data (FRED) system. It indicates the way unemployment has changed from year to year.**
# 
# **Summary: This graph shows some key indicators of U.S. unemployment from years past. Deep spikes could be times of financial crisis, and steep dips could be times of economic recovery.**

# In[ ]:


fred_data_cleaned['Date'] = pd.to_datetime(fred_data['Date'])
plt.figure(figsize=(12, 6))
sns.lineplot(data=fred_data, x='Date', y='Unemployment Rate')
plt.title('FRED Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.grid()
plt.show()


# ##  Line Chart for BLS Unemployment Data
# **Description: This plot shows decades-based unemployment data from the Bureau of Labor Statistics (BLS). It indicates the way unemployment has changed from year to year.**
# 
# **Summary: This graph shows some key indicators of U.S. unemployment from years past. Deep spikes could be times of financial crisis, and steep dips could be times of economic recovery.**

# 

# In[ ]:


#  Line Chart for BLS Unemployment Data
import matplotlib.pyplot as plt
import seaborn as sns

# Convert 'Year' to a string if necessary
bls_data_cleaned['Year'] = bls_data_cleaned['Year'].astype(str)

# Create a line plot for BLS Unemployment Data
plt.figure(figsize=(12, 6))
sns.lineplot(data=bls_data_cleaned, x='Year', y='BLS_Unemployment_Value')
plt.title('BLS Unemployment Data Over Years')
plt.xlabel('Year')
plt.ylabel('Unemployment Value')
plt.xticks(rotation=45)
plt.grid()
plt.show()



# ## Line Chart for FRED Unemployment Data
# **Description: This plot shows decades-based unemployment data from the Federal Reserve Economic Data (FRED) system. It indicates the way unemployment has changed from year to year.**
# 
# **Summary: This graph shows some key indicators of U.S. unemployment from years past. Deep spikes could be times of financial crisis, and steep dips could be times of economic recovery.**

# In[ ]:


fred_data_cleaned['Date'] = pd.to_datetime(fred_data['Date'])

# Create a line plot for FRED Unemployment Data
plt.figure(figsize=(12, 6))
sns.lineplot(data=fred_data, x='Date', y='Unemployment Rate')
plt.title('FRED Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.grid()
plt.show()


# ## Line chart for unemployment_analysis with plotly
# **Description: This interactive line chart, built with Plotly, shows unemployment rates for different countries over selected years (e.g., 2000, 2005, 2010, 2015, and 2020). It uses a dropdown or legend to allow users to toggle countries on/off, providing interactivity.**
# 
# Insights: The interactive feature enhances comparison across countries. You can easily spot which countries faced higher unemployment rates during certain years and identify global or regional trends. For example, global recessions like the 2008 financial crisis may be reflected in multiple countries.

# In[ ]:


import plotly.express as px

years_of_interest = [2000, 2005, 2010, 2015, 2020]
filtered_data = analysis_data_cleaned[analysis_data_cleaned['Year'].isin(years_of_interest)]

fig = px.line(
    filtered_data,
    x='Year',
    y='Unemployment Rate',
    color='Country Name',  # Color lines by country
    title='Interactive Unemployment Rate by Country Over Selected Years',
    labels={'Unemployment Rate': 'Unemployment Rate (%)', 'Year': 'Year'}
)

fig.update_traces(mode='lines+markers', hovertemplate='%{y}% unemployment in %{x}')
fig.update_layout(
    xaxis_title='Year',
    yaxis_title='Unemployment Rate (%)',
    hovermode='x unified',
    legend_title_text='Country',
    template='plotly_dark'  # Optional: change to 'plotly' for a lighter theme
)

fig.show()


# ## More data cleaning and merging Data BLS and FRED
# - Converted 'Year' column to datetime for BLS data
# - Merged BLS and FRED data on 'Year'

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Loading datasets
bls_data_path = 'path_to_your_bls_data.csv'  # Update this to your BLS data file path
fred_data_path = 'path_to_your_fred_data.csv'  # Update this to your FRED data file path


# Convert 'Year' column to datetime for BLS data
bls_data_cleaned['Year'] = pd.to_datetime(bls_data_cleaned['Year'])

# Convert 'Year' column in FRED data to datetime
fred_data_cleaned['Year'] = pd.to_datetime(fred_data_cleaned['Year'].astype(str) + '-01-01')

# Merge BLS and FRED data on 'Year'
bls_fred_cleaned = pd.merge(
    bls_data_cleaned[['Year', 'BLS_Unemployment_Value']],
    fred_data_cleaned[['Year', 'Unemployment Rate']],
    on='Year',
    how='outer'
)

# Plot the data
plt.figure(figsize=(12, 6))
sns.lineplot(data=bls_fred_cleaned, x='Year', y='BLS_Unemployment_Value', label='BLS Unemployment Value', color='blue')
sns.lineplot(data=bls_fred_cleaned, x='Year', y='Unemployment Rate', label='FRED Unemployment Rate', color='orange')

# Adding titles and labels
plt.title('Comparison of BLS and FRED Unemployment Rates')
plt.xlabel('Year')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)  
plt.legend()
plt.grid()
plt.tight_layout()  
plt.show()


# ## Line chart for unemployment_analysis with plotly
# **Description: Interactive Line Chart — created using PlotlyUnemployment Rates for selected Countries from 2000 to 2005, 2010, 2015, 2020. It displays countries by turning them on/off using a dropdown or legend that’s interactive.**
# 
# **Insights: Interactive option gives country comparison. It’s easy to spot which countries experienced higher unemployment in some years, and to see global or regional patterns. Global recessions such as the financial crisis of 2008, for instance, can manifest in more than one nation.**

# In[ ]:


import plotly.express as px

years_of_interest = [2000, 2005, 2010, 2015, 2020]
filtered_data = analysis_data_cleaned[analysis_data_cleaned['Year'].isin(years_of_interest)]

fig = px.line(
    filtered_data,
    x='Year',
    y='Unemployment Rate',
    color='Country Name',  
    title='Interactive Unemployment Rate by Country Over Selected Years',
    labels={'Unemployment Rate': 'Unemployment Rate (%)', 'Year': 'Year'}
)

fig.update_traces(mode='lines+markers', hovertemplate='%{y}% unemployment in %{x}')
fig.update_layout(
    xaxis_title='Year',
    yaxis_title='Unemployment Rate (%)',
    hovermode='x unified',
    legend_title_text='Country',
    template='plotly_dark'  
)

fig.show()


# ## More data cleaning and merging Data BLS and FRED
# **Description: Unemployment figures from BLS and FRED merged into one chart. It can show both datasets on separate lines or merged view with the same line so you can directly compare them.**
# 
# **Insights: This table is useful to spot any potential errors or convergence of BLS/FRED unemployment data. It’s handy for confirming that unemployment data are the same from different sources. And if both the datasets are similar it makes you feel better about the trend’s validity. If they don’t match, then that may be something to probe into reporting and/or data collection gaps.**

# 

# In[ ]:


bls_fred_cleaned.columns


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Converted 'Year' column to datetime for BLS data without time zone
bls_data_cleaned['Year'] = pd.to_datetime(bls_data_cleaned['Year']).dt.tz_localize(None)

# Converted 'Year' column in FRED data to datetime without time zone
fred_data_cleaned['Year'] = pd.to_datetime(fred_data_cleaned['Year'].astype(str) + '-01-01').dt.tz_localize(None)

# Merged BLS and FRED data on 'Year'
bls_fred_cleaned = pd.merge(
    bls_data_cleaned[['Year', 'BLS_Unemployment_Value']],
    fred_data_cleaned[['Year', 'Unemployment Rate']],
    on='Year',
    how='outer'
)

# Ploting the data
plt.figure(figsize=(12, 6))
sns.lineplot(data=bls_fred_cleaned, x='Year', y='BLS_Unemployment_Value', label='BLS Unemployment Value', color='blue')
sns.lineplot(data=bls_fred_cleaned, x='Year', y='Unemployment Rate', label='FRED Unemployment Rate', color='orange')

# Adding titles and labels
plt.title('Comparison of BLS and FRED Unemployment Rates')
plt.xlabel('Year')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)  
plt.legend()
plt.grid()
plt.tight_layout()  
plt.show()


# ## Insights and Interesting Information
# 
# So far, some thoughts are worth stating from the comparison of the unemployment data from BLS and FRED datasets:
# 
# - **Trends In Years**: If we look at the unemployment rate over the years, we see what are the trends in economic cycles such as recession or recovery. The unemployment rises in times of economic downturn, like the financial crisis of 2008 and the COVID-19 pandemic.
# 
# - **Analysis by Dataset**:The Fred dataset provides a similar trend as the BLS dataset showing accuracy in both datasets.
# 
# - **Effects of Economic Events**: Certain events (government policies, shocks (such as natural disasters)) may immediately affect unemployment rates. These impacts are visible in the data, so we can see how these kinds of incidents impact employment.
# 
# ## Distributions of Variables
# 
# - **BLS Unemployment Values**: As with most statistics in the BLS dataset, most values fall within the lower range of unemployment, while the high values are fewer in frequency. This lopsided pattern shows that high unemployment is high but less widespread in normal economies.
# 
# - **FRED Unemployment Rate**: The FRED dataset can be like that but, since it has more than one dimension, it reveals a deeper picture. Unemployment rates could be seasonal or temporally variable as a function of economic policy and society.
# 

# ## Machine learning
# For this project I could try various machine learning methods like supervised learning methods. I might use regression to predict unemployment, for example, on the basis of some critieria such as economic metrics or demographic information. Classification Algorithms could also be applicable if I am looking to sort data according to regions of high, medium, or low unemployment.
# 
# But there are a couple of problems here. One is the quality and completeness of the data. Data that’s missing or not reflected can bias or make your model fail. Also, if the dataset is disproportionately large (e.g., more data points are in some years/locations) this can affect classification accuracy.
# 
# Features and engineering are a second possibility. It is very important for performance to decide which features to map on to the model and how to model them. And there are also sometimes complex relationships in the data that need more complex models or preprocessing.

# # Exploratory Data Analysis (EDA)
# To start, I am going to check the datasets to see how it is structured and for missing values or out of order time spans. I will plot the data as line plots, histograms, and scatter plots to look for pattern and exceptions in the unemployment rates.

# In[68]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# # Merging the data

# In[71]:


bls_data = pd.read_csv('extracted_files/bls_unemployment_data.csv')
fred_data = pd.read_csv('extracted_files/fred_unemployment_data.csv')
analysis_data = pd.read_csv('extracted_files/unemployment_analysis.csv')

bls_data_cleaned = bls_data.drop(columns=['Period', 'Footnote']).rename(columns={'Year': 'Year', 'Value': 'BLS_Unemployment_Value'})
fred_data_cleaned = fred_data.rename(columns={'Date': 'Year', 'Unemployment Rate': 'FRED_Unemployment_Rate'})
analysis_data_cleaned = analysis_data.drop(columns=['Country Code']).rename(columns={'Country Name': 'Country'}).melt(id_vars=['Country'], var_name='Year', value_name='Country_Unemployment_Rate')

bls_data_cleaned['Year'] = pd.to_numeric(bls_data_cleaned['Year'], errors='coerce')
fred_data_cleaned['Year'] = pd.to_numeric(fred_data_cleaned['Year'], errors='coerce')
analysis_data_cleaned['Year'] = pd.to_numeric(analysis_data_cleaned['Year'], errors='coerce')

merged_data = pd.merge(bls_data_cleaned, fred_data_cleaned, on='Year', how='outer')
merged_data = pd.merge(merged_data, analysis_data_cleaned, on='Year', how='outer')

imputer = SimpleImputer(strategy='mean')
merged_data_imputed = pd.DataFrame(imputer.fit_transform(merged_data.select_dtypes(include=['float64', 'int64'])), columns=merged_data.select_dtypes(include=['float64', 'int64']).columns)
final_data = pd.concat([merged_data.select_dtypes(exclude=['float64', 'int64']), merged_data_imputed], axis=1)


# # Splitting the Dataset
# I’ll split the dataset into training and test data with previous years of training data and later years of test data for the purpose of prediction. This helps to keep the time order consistent across time series.

# In[72]:


X = final_data.drop(['Country'], axis=1)
y = final_data['Country']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


# # Data Scaling and Normalization
# For features sensitive models like KNN or SVM, I will use StandardScaler to get the numerical data as close as possible so that all features are of equal size and do not overscale the mode

# # Handling Categorical Data and adding Data scaling and normalization 
# When we have categorical variables, like country names in the dataset, I encode those using OneHotEncoding or LabelEncoder to support Machine Learning algorithms. I also will use StandardScaler to get the numerical data as close as possible so that all features are of equal size and do not overscale the mode

# In[73]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder



# Encode the target variable if it's categorical
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Separate numeric and categorical features
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Define separate transformers for scaling and normalization of numeric features
scaling_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
    ('scaler', StandardScaler())  # Standardization (scaling)
])

normalization_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
    ('normalizer', MinMaxScaler())  # Min-Max Normalization
])

# Preprocessor to apply scaling to certain columns and normalization to others
preprocessor = ColumnTransformer(
    transformers=[
        ('scale', scaling_transformer, numeric_features),  # Scaling
        ('normalize', normalization_transformer, numeric_features)  # Normalization
    ]
)

# Model pipeline integrating preprocessor and classifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])


# # Split the data into training and testing sets

# In[74]:


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)
accuracy = model_pipeline.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")


# # Testing Multiple Algorithms
# I will try some different ML algorithms and see which is the best. These can be Linear Regression, Random Forest, SVM etc. I’ll run all the models, and assess the performance in MSE and R-squared metrics (regression models), or accuracy for classification models.

# In[75]:


models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC()
}

for model_name, model in models.items():
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
  
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}")


# Random Forest seems to be the most accurate model

# Havent recieved any feedback yet.

# ## Resources and References
# *What resources and references have you used for this project?*
# 📝
# - Lecture videos
# - https://seaborn.pydata.org/tutorial/introduction.html
# - https://builtin.com/data-science/data-visualization-tutorial
# - https://matplotlib.org/stable/users/explain/quick_start.html
# - https://plotly.com/python/getting-started/

# In[ ]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

