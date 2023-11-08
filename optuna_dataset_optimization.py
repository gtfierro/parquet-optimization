#!/usr/bin/env python
# coding: utf-8

# In[1]:



# In[2]:


import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import pandas as pd
from itertools import product
import time
import tempfile
import os
import optuna
from joblib import parallel_backend


# In[3]:


df = pd.read_csv('vav_data.csv')
df.columns = ['time','value','building','type']
df.sort_values(by=['type','time'], inplace=True)
table = pa.Table.from_pandas(df)


# In[4]:


def objective(trial):
    # we define the parameter space we want each trial to select from
    row_group_size = trial.suggest_int('row_group_size', 1024*1024*1024, 10*1024*1024*1024, step=1024*1024)
    data_page_size = trial.suggest_int('data_page_size', 1024*1024*1024, 10*1024*1024*1024, step=1024*1024)

    # Create a dictionary to hold the compression algorithm for each column.
    columns = df.columns
    column_compressions = {}
    dictionary_columns = []
    for column in columns:
        # Suggest a compression algorithm for each column.
        column_compressions[column] = trial.suggest_categorical(f'compression_{column}', ['NONE', 'SNAPPY', 'GZIP', 'BROTLI', 'LZ4', 'ZSTD'])

        # Use Optuna to decide whether to include each column in the list
        include_column = trial.suggest_categorical(f'dictionary_encode_{column}', [True, False])
        if include_column:
            dictionary_columns.append(column)
    
    
    # Use a temporary directory to write the output.parquet file
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = os.path.join(tmpdirname, "output.parquet")
        pq.write_table(table, filepath, row_group_size=row_group_size, compression=column_compressions, data_page_size=data_page_size, use_dictionary=dictionary_columns)
        filesize = os.path.getsize(filepath)

        # Read the parquet file into a pandas DataFrame
        df_output = pd.read_parquet(filepath)
        
        t0 = time.time()

        # Perform the groupby operation and calculate the min, max, and average
        aggregated_data = df_output.groupby(['building', 'type'])['value'].agg(['min', 'max', 'mean'])
        
    return time.time() - t0, filesize


# In[ ]:


# Create a study object and specify the optimization direction ('minimize' or 'maximize').
storage = optuna.storages.RDBStorage(url="sqlite:///optuna.sqlite3", engine_kwargs={"connect_args": {"timeout": 100}})
study = optuna.create_study(storage=storage, directions=['minimize','minimize'])

# Optimize the study, the objective function is passed in as the first argument.
study.optimize(objective, n_trials=10000, n_jobs=16)

# storage = "sqlite:///optuna.db"
# study = optuna.create_study(storage=storage)

# with parallel_backend('multiprocessing'):  # Overrides `prefer="threads"` to use multi-processing.
#     study.optimize(objective, n_trials=30, n_jobs=8)


# In[1]:


# Results
print('Number of finished trials:', len(study.trials))
for trial in study.best_trials:
    print(trial.params)
    print(trial.values)


# In[2]:
