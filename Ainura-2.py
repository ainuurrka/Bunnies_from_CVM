#!/usr/bin/env python
# coding: utf-8

# In[99]:


import pandas as pd


# In[100]:


train_amplitude_0 = pd.read_parquet('train_amplitude_chunk_00.parquet', engine='pyarrow')


# In[101]:


train_amplitude_1 = pd.read_parquet('train_amplitude_chunk_01.parquet', engine='pyarrow')


# In[102]:


train_amplitude_2 = pd.read_parquet('train_amplitude_chunk_02.parquet', engine='pyarrow')


# In[103]:


train_amplitude_0


# In[90]:


train_amplitude


# In[91]:


def generate_antifraud_features_with_time(train_amplitude):
    train_amplitude = train_amplitude.copy()

    # 
    train_amplitude['application_date'] = pd.to_datetime(train_amplitude['application_date'], errors='coerce')

    # Временные фичи
    train_amplitude['application_hour'] = train_amplitude['application_date'].dt.hour
    train_amplitude['application_weekday'] = train_amplitude['application_date'].dt.dayofweek
    train_amplitude['is_night'] = train_amplitude['application_hour'].between(0, 5).astype(int)
    train_amplitude['is_weekend'] = train_amplitude['application_weekday'].isin([5, 6]).astype(int)

    # Основные агрегаты
    agg_funcs = {
        'device_model': lambda x: x.nunique(),
        'device_type': lambda x: x.nunique(),
        'os_name': lambda x: x.nunique(),
        'platform': lambda x: x.nunique(),
        'version_name': lambda x: x.nunique(),
        'ip_address': lambda x: x.nunique(),
        'region': lambda x: x.nunique(),
        'language': lambda x: x.nunique(),
        'location_lat': lambda x: x.nunique(),
        'location_lng': lambda x: x.nunique(),
        'event_type': lambda x: x.nunique(),
        'event_id': 'count',
        'session_id': lambda x: x.nunique(),
        'user_id': lambda x: x.nunique(),
        'user_properties': lambda x: (x.astype(str) == '{}').mean(),
        'is_night': 'mean',
        'is_weekend': 'mean'
    }

    rename_dict = {
        'device_model': 'n_device_models',
        'device_type': 'n_device_types',
        'os_name': 'n_os_names',
        'platform': 'n_platforms',
        'version_name': 'n_versions',
        'ip_address': 'n_ips',
        'region': 'n_regions',
        'language': 'n_languages',
        'location_lat': 'n_latitudes',
        'location_lng': 'n_longitudes',
        'event_type': 'n_event_types',
        'event_id': 'event_count',
        'session_id': 'n_sessions',
        'user_id': 'n_user_ids',
        'user_properties': 'share_empty_user_properties',
        'is_night': 'share_night',
        'is_weekend': 'share_weekend'
    }

    agg_df = train_amplitude.groupby('applicationid').agg(agg_funcs).rename(columns=rename_dict).reset_index()
    return agg_df


# In[105]:


features_df_0 = generate_antifraud_features_with_time(train_amplitude_0)


# In[107]:


features_df_1 = generate_antifraud_features_with_time(train_amplitude_1)


# In[108]:


features_df_2 = generate_antifraud_features_with_time(train_amplitude_2)


# In[109]:


combined = pd.concat([features_df_0, features_df_1, features_df_2], ignore_index=True)


# In[110]:


combined


# In[111]:


unique_count = combined['applicationid'].nunique()

print(f'Уникальное количество applicationid: {unique_count}')


# In[93]:


features_df


# In[94]:


features_df.to_csv('train_amplitude_chunk_02_ainura.csv', index=False, sep=',', encoding='utf-8')


# In[87]:


train_amplitude['applicationid'].nunique()


# In[71]:


unique_device_models = train_amplitude['device_model'].dropna().unique()


# In[73]:


unique_device_models[:30]


# In[74]:


train_amplitude


# In[75]:


train_amplitude['device_model'].isna().sum()


# In[17]:


train_amplitude.columns


# In[ ]:





# In[112]:


valid_amplitude = pd.read_parquet('valid_amplitude_chunk_00.parquet', engine='pyarrow')


# In[113]:


valid_amplitude


# In[117]:


valid_amplitude.columns


# In[118]:


def generate_antifraud_features_with_time(valid_amplitude):
    valid_amplitude = valid_amplitude.copy()

    # 
    valid_amplitude['application_date'] = pd.to_datetime(valid_amplitude['application_date'], errors='coerce')

    # Временные фичи
    valid_amplitude['application_hour'] = valid_amplitude['application_date'].dt.hour
    valid_amplitude['application_weekday'] = valid_amplitude['application_date'].dt.dayofweek
    valid_amplitude['is_night'] = valid_amplitude['application_hour'].between(0, 5).astype(int)
    valid_amplitude['is_weekend'] = valid_amplitude['application_weekday'].isin([5, 6]).astype(int)

    # Основные агрегаты
    agg_funcs = {
        'device_model': lambda x: x.nunique(),
        'device_type': lambda x: x.nunique(),
        'os_name': lambda x: x.nunique(),
        'platform': lambda x: x.nunique(),
        'version_name': lambda x: x.nunique(),
        'ip_address': lambda x: x.nunique(),
        'region': lambda x: x.nunique(),
        'language': lambda x: x.nunique(),
        'location_lat': lambda x: x.nunique(),
        'location_lng': lambda x: x.nunique(),
        'event_type': lambda x: x.nunique(),
        'event_id': 'count',
        'session_id': lambda x: x.nunique(),
        'user_id': lambda x: x.nunique(),
        'user_properties': lambda x: (x.astype(str) == '{}').mean(),
        'is_night': 'mean',
        'is_weekend': 'mean'
    }

    rename_dict = {
        'device_model': 'n_device_models',
        'device_type': 'n_device_types',
        'os_name': 'n_os_names',
        'platform': 'n_platforms',
        'version_name': 'n_versions',
        'ip_address': 'n_ips',
        'region': 'n_regions',
        'language': 'n_languages',
        'location_lat': 'n_latitudes',
        'location_lng': 'n_longitudes',
        'event_type': 'n_event_types',
        'event_id': 'event_count',
        'session_id': 'n_sessions',
        'user_id': 'n_user_ids',
        'user_properties': 'share_empty_user_properties',
        'is_night': 'share_night',
        'is_weekend': 'share_weekend'
    }

    agg_df = valid_amplitude.groupby('applicationid').agg(agg_funcs).rename(columns=rename_dict).reset_index()
    return agg_df


# In[120]:


features_df_valid = generate_antifraud_features_with_time(valid_amplitude)


# In[121]:


valid_amplitude


# In[122]:


features_df_valid


# In[124]:


features_df_valid.to_csv('valid_amplitude.csv', index=False, sep=',', encoding='utf-8')


# In[123]:


valid_amplitude['device_model'].isna().sum()


# In[125]:


features_df_valid


# In[ ]:




