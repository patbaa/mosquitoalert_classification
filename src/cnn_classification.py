# train a ResNet50 CNN with yearly cross-validation
# and generate predictions with it
################################################################################
# imports

import pandas as pd
from utils import *

################################################################################
# load metadata

data_folder = '/home/pataki/dummy_mosquito/mosquitoAlert-master'
meta = pd.read_csv(f'{data_folder}/meta_all_raw.csv')


################################################################################
# training only on mosquito alert data

all_preds  = []
all_labels = []
all_years  =  []
for y in [2014, 2015, 2016, 2017, 2018, 2019]:
    years = meta[meta.year == y].year.tolist()    
    preds, labels = train_predict(df_train = meta[meta.year != y][['file', 'isTiger']], 
                                  df_test =  meta[meta.year == y][['file', 'isTiger']], 
                                  path='/')
        
    all_preds  += preds
    all_labels += labels
    all_years  += years
    
orig = pd.DataFrame({'year':all_years,
                     'pred':np.array(all_preds)[:,1],
                     'label':all_labels})

orig.to_csv('only_mosquitoalert_results.csv', index=False)

################################################################################
# training on augmented data
ip102 = pd.read_csv(f'{data_folder}/ip102_aug.csv')
ip102['isTiger'] = False
ip102['year'] = -1
ip102['file'] = ip102.img
aug = meta[['year', 'isTiger', 'file']].append(ip102[['year', 'isTiger', 'file']])

aug_preds  = []
aug_labels = []
aug_years  =  []
for y in [2014, 2015, 2016, 2017, 2018, 2019]:
    years = aug[aug.year == y].year.tolist()    
    preds, labels = train_predict(df_train = aug[aug.year != y][['file', 'isTiger']], 
                                  df_test =  aug[aug.year == y][['file', 'isTiger']], 
                                  path='/')
        
    aug_preds  += preds
    aug_labels += labels
    aug_years  += years
    
aug = pd.DataFrame({'year':aug_years,
                    'pred':np.array(aug_preds)[:,1],
                    'label':aug_labels})

aug.to_csv('augmented_results.csv', index=False)