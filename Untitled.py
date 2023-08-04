#!/usr/bin/env python
# coding: utf-8

# # Steps followed
# 1.Importing necessary libraries
# 2.Creating S3 bucket
# 3.Mapping train and test data in S3
# 4.mapping the path of the models in S3

# In[2]:


import sagemaker
import boto3
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.session import s3_input , Session


# In[13]:


bucket_name='mybuckettttttttt'
my_region=boto3.session.Session().region_name
print(my_region)


# In[14]:


s3=boto3.resource('s3')
try:
    if my_region=='ap-south-1':
        s3.create_bucket(Bucket=bucket_name)
    print('s3 bucket create successfully')
except Exception as e:
    print('S3 error: ',e)


# In[15]:


# set an output path where the trained model will be saved
prefix='xgboost-as-a-built-in-algo'
output_path='s3://{}/{}/output'.format(bucket_name,prefix)
print(output_path)


# Downloading the dataset and storing in S3

# In[25]:


import pandas as pd
import urllib
try:
    urllib.request.urlretrieve ("https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv", "bank_clean.csv")
    print('Success: downloaded bank_clean.csv.')
except Exception as e:
    print('Data load error: ',e)
try:
    model_data=pd.read_csv('./bank_clean.csv',index_col=0)
    print("Success : Data loaded into dataframe.")
except Exception as e:
    print('Data load error :',e)


# In[26]:


import numpy as np
train_data,test_data=np.split(model_data.sample(frac=1,random_state=1729),[int(0.7*len(model_data))])
print(train_data.shape,test_data.shape)


# In[27]:


model_data.head()


# In[32]:


from sagemaker.inputs import TrainingInput


# In[34]:


import os
pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], 
                                                axis=1)], 
                                                axis=1).to_csv('train.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
TrainingInput_train = sagemaker.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')


# In[35]:


pd.concat([test_data['y_yes'], test_data.drop(['y_no', 'y_yes'], 
                                                axis=1)], 
                                                axis=1).to_csv('test.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'test/test.csv')).upload_file('test.csv')
TrainingInput_test = sagemaker.TrainingInput(s3_data='s3://{}/{}/test'.format(bucket_name, prefix), content_type='csv')


# Building Models Xgboot

# In[58]:


container = image_uris.retrieve(region=boto3.Session().region_name, framework='xgboost', version='1.0-1')


# In[65]:


hyperparameters={
    "max_depth": "5",
    "eta":"0.2",
    "gamma":"4",
    "min_child_weight":"6",
    "subsample":"0.7",
    "objective":"binary:logistic",
    "num_round":50
}


# In[52]:


import sagemaker.image_uris as image_uris


# In[66]:


# construct a SageMaker estimator that calls the xgboost-container
estimator = sagemaker.estimator.Estimator(image_uri=container, 
                                          hyperparameters=hyperparameters,
                                          role=sagemaker.get_execution_role(),
                                          instance_count=1, 
                                          instance_type='ml.m5.2xlarge', 
                                          volume_size=5, # 5 GB 
                                          output_path=output_path,
                                          use_spot_instances=True,
                                          max_run=300,
                                          max_wait=600)


# In[67]:


estimator.fit({'train':TrainingInput_train,'validation': TrainingInput_test})


# Deploy M.L. Model

# In[69]:


xgb_predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')


# Prediction of Test Data

# In[72]:


from sagemaker.serializers import CSVSerializer
csv_serializer=CSVSerializer()


# In[76]:


xgb_predictor.content_type = 'text/csv'
xgb_predictor.serializer = csv_serializer

predictions = xgb_predictor.predict(test_data_array)
predictions_array = np.fromstring(predictions[1:], sep=',')
print(predictions_array.shape)


# In[77]:


predictions_array


# In[78]:


cm = pd.crosstab(index=test_data['y_yes'], columns=np.round(predictions_array), rownames=['Observed'], colnames=['Predicted'])
tn = cm.iloc[0,0]; fn = cm.iloc[1,0]; tp = cm.iloc[1,1]; fp = cm.iloc[0,1]; p = (tp+tn)/(tp+tn+fp+fn)*100
print("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
print("{0:<15}{1:<15}{2:>8}".format("Predicted", "No Purchase", "Purchase"))
print("Observed")
print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format("No Purchase", tn/(tn+fn)*100,tn, fp/(tp+fp)*100, fp))
print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format("Purchase", fn/(tn+fn)*100,fn, tp/(tp+fp)*100, tp))


# Deleting the Endpoints

# In[ ]:


sagemaker.Session().delete_endpoint(xgb_predictor.endpoint)
bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
bucket_to_delete.objects.all().delete()

