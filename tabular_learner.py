import time
from fastai.tabular.all import *

def run_tabular_learner():
    data_path = '/opt/ml/input/data/'

    train_df = data['train']
    test_df = data['test']

    batch_size = 64
    valid_pct = 0.2
    random_seed = 42
    cycles=1
    categorical_features = ['user_id', 'isbn', 'location_city', 'location_state',\
                            'location_country', 'publisher', 'category', 'category_high',\
                            'language', 'book_author', 'book_title', 'summary']
    continuous_features = ['age', 'year_of_publication']
    target='rating'

    # init dataset
    splits = RandomSplitter(valid_pct=valid_pct, seed=random_seed)(range_of(df))
    to = TabularPandas(train_df, 
                       procs=[Categorify, FillMissing, Normalize],
                       cat_names=categorical_features,
                       cont_names=continuous_features,
                       y_names=target,
                       splits=splits)

    # init dataloader
    dls = to.dataloaders(bs=batch_size)

    # init learner
    learn = tabular_learner(dls, metrics=rmse)

    # train
    learn.fit_one_cycle(cycles)

    # test
    dataloader = learn.dls.test_dl(test_df)
    predicts = learn.get_preds(dl=dataloader)
    ratings = predicts[0]

    # save predicted ratings as "tabular_learner{time}_{}"
    now = time.localtime()
    now_date = time.strftime('%Y%m%d', now)
    now_hour = time.strftime('%X', now)
    save_time = now_date + '_' + now_hour.replace(':', '')
    submission = pd.read_csv(data_path + "sample_submission.csv")
    submission['rating'] = ratings
    submission.to_csv('submit/{}_{}.csv'.format(save_time, 'tabularLearner'), index=False)


if __name__ == "__main__":
    print(f'--------------- tabular learner ---------------')
    run_tabular_learner()