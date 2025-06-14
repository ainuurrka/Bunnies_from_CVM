#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pandas as pd
import pickle
loaded_model = pickle.load(open('Final_Model.sav', 'rb'))
dfr = pd.read_csv('validation datacet.csv')
predictions = loaded_model.predict_proba(dfr[loaded_model.feature_names_in_])[:,1]
pred_df = pd.DataFrame(data=predictions, columns =['predicted_probability'])
result_with_probas = pd.concat([dfr[['APPLICATIONID']],pred_df], ignore_index=True,axis=1)
result_with_probas.columns = ['id','score']
result_with_probas = result_with_probas.sort_values(by=['score'], ascending=False)
result_with_probas['isFrod'] = False
result_with_probas.loc[result_with_probas['score'] >= 0.238416, 'isFrod'] = True
result_with_probas.to_csv('result.csv',index=False)

