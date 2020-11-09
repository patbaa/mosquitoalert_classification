################################################################################
# imports

import pandas as pd
from PIL import Image
from pathlib import Path
from collections import Counter

################################################################################
# identify files

## folder where http://gitlab.ceab.csic.es/jgarriga/mosquitoAlert is downloaded
## first you need to extract tarballs, I've done it manually
data_folder = '/home/pataki/mosquito/data'

files2014 = list(Path(f'{data_folder}/all_raw/2014/').glob('*png'))
files2015 = list(Path(f'{data_folder}/all_raw/2015/').glob('*png'))
files2016 = list(Path(f'{data_folder}/all_raw/2016/').glob('*png'))
files2017 = list(Path(f'{data_folder}/all_raw/2017/').glob('*png'))
files2018 = list(Path(f'{data_folder}/all_raw/2018/').glob('*png'))
files2019 = list(Path(f'{data_folder}/all_raw/2019/').glob('*png'))

################################################################################
# reorganize file structure, all years data to a single folder

Path(f'{data_folder}/all_raw/all').mkdir(parents=True, exist_ok=True)

for i in files2014:
    fname = i.stem
    i.rename(f'{data_folder}/all_raw/all/2014_{fname}.png')
    
for i in files2015:
    fname = i.stem
    i.rename(f'{data_folder}/all_raw/all/2015_{fname}.png')
    
for i in files2016:
    fname = i.stem
    i.rename(f'{data_folder}/all_raw/all/2016_{fname}.png')
    
for i in files2017:
    fname = i.stem
    i.rename(f'{data_folder}/all_raw/all/2017_{fname}.png')
    
for i in files2018:
    fname = i.stem
    i.rename(f'{data_folder}/all_raw/all/2018_{fname}.png')
    
for i in files2019:
    fname = i.stem
    i.rename(f'{data_folder}/all_raw/all/2019_{fname}.png')
    
################################################################################
# create new metadata
    
df = pd.DataFrame()
for y in ['2014', '2015', '2016', '2017', '2018', '2019']:
    tmp = pd.read_csv(f'{data_folder}/all_raw/{y}/imgRef.txt')
    tmp['imgNmb'] = [y + '_' + str(imnb).zfill(6) + '.png' for 
                     imnb in tmp.imgNmb.values]
    df = df.append(tmp)
df.to_csv(f'{data_folder}/all_raw/all/all_imgRef.txt', index=False)

################################################################################
# select relevant data for classification

meta = pd.read_csv(f'{data_folder}/all_raw/all/all_imgRef.txt')
meta = meta[(~meta.hidden) & 
            (meta.imgLabel != 'notClassified') & 
            (meta.imgClass != 'canNotTell')]

meta = meta[~meta.imgClass.isin(['otherSites', 'site'])]
meta = meta[(~meta.hidden) & 
            (meta.imgLabel != 'notClassified') & 
            (meta.imgClass != 'canNotTell')]
meta = meta[~meta.imgClass.isin(['otherSites', 'site'])]

meta['year']  = [i.split('_')[0] for i in meta.imgNmb]
meta['file'] = [f'{data_folder}/all_raw/all/' + i for i in meta.imgNmb]
meta['isTiger'] = [int(i == 'Ae.albopictus') for i in meta.imgClass]
meta.to_csv(f'{data_folder}/meta_all_raw.csv', index=False)

################################################################################
# add a part of IP102 dataset for augmentation
ip102_folder = '/media/nagy/IP102/ip102_v1.1'

IP102 = pd.read_csv(f'{ip102_folder}/train.txt', sep=' ', 
                    names=['img', 'class'])
IP102['class'] = IP102['class'].values + 1

IP102 = IP102[IP102['class'].isin([6, 8, 12, 29, 34, 36, 37, 38, 41, 
                                   47, 51, 66, 85, 86])]
IP102['img'] = [f'{ip102_folder}/images/' + i for i in IP102.img.values]
IP102.to_csv(f'{data_folder}/ip102_aug.csv', index=False)