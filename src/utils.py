from fastai.vision import *
from fastai.metrics import error_rate
        
def train_predict(df_train, df_test, path,
          bs = 32, img_s = 256, n_epoch1 = 5, n_epoch2 = 5,
          maxlr2 = 5e-5):
    # df_train and df_test should contain
    ## isTiger column
    ## file col with absolute path to the img
    training = (ImageList.from_df(df_train, path=path).
                split_none().
                label_from_df(cols='isTiger', label_cls = CategoryList))
    
    training_data = (training.transform(get_transforms(),size=img_s).
            databunch(bs=bs).normalize(imagenet_stats))
    
    
    test = (ImageList.from_df(df_test, path=path).
             split_none().
             label_from_df(cols='isTiger', label_cls = CategoryList))
    
    test_data = (test.transform(get_transforms(),size=img_s).
            databunch(bs=bs).normalize(imagenet_stats))
    
    learn = cnn_learner(training_data, models.resnet50, metrics=error_rate)
    learn.fit_one_cycle(n_epoch1)
    learn.unfreeze()
    learn.fit_one_cycle(n_epoch1, max_lr=maxlr2)
    
    # generate predictions
    preds = []
    labels = []
    for i in range(len(test_data.train_ds)):
        p = learn.predict(test_data.train_ds.x[i])
        preds.append(list(np.array(p[2])))
        labels.append(int(test_data.train_ds.y[i]))
        
    return preds, labels