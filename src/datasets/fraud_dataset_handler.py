from cmath import isnan
from multiprocessing import allow_connection_pickling
import pandas as pd
import numpy as np


def sample_data(samples):
    """
    Function to save some samples from the data so we can discard the data we
    dont need

    Args:  
            samples(int):   The number of instances we want to extract from the dataset

    Return: (array):    The number of sampels we want
    """
    df=pd.read_csv('card_transaction.csv')
    df.drop(['User','Card','Year','Month','Day','Merchant Name','Merchant City','Merchant State','Errors?', 'Use Chip'],axis=1, inplace=True)
    #df=df.dropna(how='all')
    df.dropna(subset = ["Zip"], inplace=True)

    numpy_df=df.to_numpy()

    fraud_index=np.where(numpy_df.T[-1]=='Yes')[0]
    notfraud_index=np.where(numpy_df.T[-1]=='No')[0]
    
    random_fraudsamples=np.random.choice(fraud_index, int(samples*0.2), replace=False)
    random_notfraudsamples=np.random.choice(notfraud_index, int(samples*0.8), replace=False)

    random_samples=np.sort(np.concatenate((random_fraudsamples, random_notfraudsamples)))
    df_numpy=numpy_df[random_samples]

    
    np.save('time_amount_zip_mcc', df_numpy)

    return df_numpy

def transform_dataset(sample_list):
    """
    Discretizing the dataset(Probably a more efficient 
    way to do this, but it does the job)
    
    Args:  
        sample(list): Instance of credit card data

    Return:
        Discretized data sample
    """
    #Time 
    if int(sample_list[0][0:2])<11:
        sample_list[0]=0

    elif int(sample_list[0][0:2])>18:
        sample_list[0]=2
    else:
        sample_list[0]=1

    #Amount
    if float(sample_list[1][1:])<50:
        sample_list[1]=0
    elif float(sample_list[1][1:])>150:
        sample_list[1]=2
    else:
        sample_list[1]=1

    #ZIP
    #West=9xxx,8xxx and 59xx
    if int(sample_list[2])>=80000 or int(str(sample_list[2])[0:2])==59:
        sample_list[2]=2
    #East=0xxx-4xxx
    elif int(sample_list[2])<=40000:
        sample_list[2]=0
    else:
        sample_list[2]=1

    #MCC
    mcc_cat=np.zeros(10)
    mcc_cat[int(str(sample_list[3])[0])]=1

    #print(sample_list[3], mcc_cat)
    #merge the list into the features
    sample_list=np.delete(sample_list, -2)
    sample_list=np.insert(sample_list, 3, mcc_cat)

    #1=Fraud, 0=not fraud
    if sample_list[-1]=='Yes':
        sample_list[-1]=1
    else:
        sample_list[-1]=0
    
    return sample_list


#Just a function to shorten the data. The dataset we sample from is so big that it wont upload to
#github, we sample 1000 instances. The dataset can be found here: https://www.kaggle.com/paulrohan2020/#synthetic-transactions-virtual-world-simulator

#data=sample_data(1000)

#Now we preprocess the data:
data=np.load('time_amount_zip_mcc.npy', allow_pickle=True)
dataset=np.zeros((len(data), len(data[0])+9))
for i in range(len(data)):
    dataset[i]=transform_dataset(data[i])


np.save('time_amount_zip_mcc_1000_instances', dataset)



