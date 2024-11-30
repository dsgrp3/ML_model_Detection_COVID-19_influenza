#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pycaret
from pycaret.classification import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r"T:\my download BAG\ILIdata.csv", index_col=0)  
df.columns=df.columns.str.strip()
df.head()


# In[10]:


df.describe()


# In[11]:


df.columns


# In[12]:


sns.heatmap(df.corr(),annot=True,cmap='coolwarm',linewidths=0.4)


# In[14]:


ili_data_encoded = pd.get_dummies(ili_data, drop_first=True)
plt.figure(figsize=(12, 8))
sns.heatmap(ili_data_encoded.corr(), annot=True, cmap='coolwarm', linewidths=0.4)
plt.title("Correlation Matrix (Encoded)")
plt.show()


# In[7]:


# Import necessary libraries
import pandas as pd
from pycaret.classification import *  # Use 'from pycaret.regression import *' for regression tasks

# Load your dataset
df = pd.read_csv(r"T:\my download BAG\ILIdata.csv")  # Update the path to your dataset

# Optional: Display the first few rows of the dataset
print(df.head())

# Set up the PyCaret environment
# Specify the target variable (replace 'target' with your actual target column name)
clf = setup(data=df, target='Covid_19_result', session_id=123)

# Compare different models to find the best one
best_model = compare_models()

# Create a Random Forest model
rf_model = create_model('rf')

# Tune the Random Forest model (optional)
tuned_rf_model = tune_model(rf_model)

# Evaluate the model
evaluate_model(tuned_rf_model)

# Finalize the model
final_model = finalize_model(tuned_rf_model)

# Make predictions on new data (if you have a new dataset)
# new_data = pd.read_csv(r"T:\my download BAG\new_data.csv")  # Load new data if available
# predictions = predict_model(final_model, data=new_data)

# Save the model for future use
save_model(final_model, 'rf_model')

# Optional: Load the model later
# loaded_model = load_model('rf_model')


# In[ ]:




