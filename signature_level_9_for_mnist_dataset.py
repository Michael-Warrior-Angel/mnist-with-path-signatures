#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import numpy as np
import pandas as pd
import iisignature
from tensorflow.keras.datasets import mnist


# In[2]:


# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# In[3]:


# Function to compute path signatures for a given level
def compute_signatures(images, signature_level):
    signatures = []
    for image in images:
        coords = [(x, y) for x in range(image.shape[0]) for y in range(image.shape[1]) if image[x, y] > 0]
        path = np.array(coords)
        signature = iisignature.sig(path, signature_level)
        signatures.append(signature)
    return np.array(signatures)


# In[4]:


# Set the signature level
signature_level = 9


# In[5]:


# Compute path signatures for train and test sets
print("Computing path signatures...")
# Measure time to compute coordinates and path signature
start_time = time.time()
train_signatures = compute_signatures(train_images, signature_level)
test_signatures = compute_signatures(test_images, signature_level)
computation_time = time.time() - start_time
print(f"Time taken to compute signatures for level {signature_level}: {computation_time:.2f} seconds")
print("Path signature computation complete.")


# In[6]:


# Concatenate labels with signatures
train_signatures_with_labels = np.column_stack((train_labels, train_signatures))
test_signatures_with_labels = np.column_stack((test_labels, test_signatures))


# ## To save the computed signatures along with labels to CSV (first column is the label).

# In[7]:


# Save the computed signatures along with labels to CSV
print("Saving signatures with labels to CSV...")
pd.DataFrame(train_signatures_with_labels).to_csv("train_signatures_level_9_with_labels.csv", index=False, header=False)
pd.DataFrame(test_signatures_with_labels).to_csv("test_signatures_level_9_with_labels.csv", index=False, header=False)
print("Signatures and labels saved successfully.")


# ## To load the saved CSV for further data processing ...

# In[8]:


# To load the saved data for future use
print("Loading signatures from CSV...")
train_signatures_loaded = pd.read_csv("train_signatures_level_9_with_labels.csv", header=None).values
test_signatures_loaded = pd.read_csv("test_signatures_level_9_with_labels.csv", header=None).values


# In[9]:


# Separate labels from signatures (first column is label)
train_labels_loaded = train_signatures_loaded[:, 0]
test_labels_loaded = test_signatures_loaded[:, 0]
train_signatures_loaded = train_signatures_loaded[:, 1:]
test_signatures_loaded = test_signatures_loaded[:, 1:]
print("Signatures and labels loaded successfully.")


# ## To standardize the signatures

# In[11]:


from sklearn.preprocessing import StandardScaler

# Initialize StandardScaler
scaler = StandardScaler()


# In[12]:


# Apply StandardScaler to the path signature columns
train_signatures_scaled = scaler.fit_transform(train_signatures_loaded)
test_signatures_scaled = scaler.transform(test_signatures_loaded)


# In[13]:


# Combine the standardized signatures with the labels
train_signatures_scaled_with_labels = np.column_stack((train_labels_loaded, train_signatures_scaled))
test_signatures_scaled_with_labels = np.column_stack((test_labels_loaded, test_signatures_scaled))


# In[14]:


# Optional: Convert back to DataFrame for better visualization or saving
train_df = pd.DataFrame(train_signatures_scaled_with_labels)
test_df = pd.DataFrame(test_signatures_scaled_with_labels)


# ## To save the standardized signatures to csv

# In[15]:


# Save the standardized signatures to csv
train_df.to_csv("train_signatures_scaled_with_labels.csv", index=False, header=False)
test_df.to_csv("test_signatures_scaled_with_labels.csv", index=False, header=False)
print("Standardization complete and data saved.")

