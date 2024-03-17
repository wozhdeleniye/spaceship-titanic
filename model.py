import pandas as pd


import os
import pickle
import argparse
import logging

from sklearn.preprocessing import LabelEncoder
from catboost import Pool, CatBoostClassifier

def replace_mode(table, column):
    val = table[column].mode()[0]
    table[column] = table[column].fillna(val)


def replace_median(table, column):
    val = table[column].median()
    table[column] = table[column].fillna(val)

def replace_empty_cabin(table, column):
    val = 'Z/9999/Z'
    table[column] = table[column].fillna(val)

def format_data():
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    combine = [train_df, test_df]
    pd.set_option('future.no_silent_downcasting', True)

    for dataset in combine:
        replace_median(dataset, 'Age')
        replace_median(dataset, 'RoomService')
        replace_median(dataset, 'FoodCourt')
        replace_median(dataset, 'ShoppingMall')
        replace_median(dataset, 'Spa')
        replace_median(dataset, 'VRDeck')

        replace_mode(dataset, 'HomePlanet')
        replace_mode(dataset, 'CryoSleep')
        replace_mode(dataset, 'Destination')
        replace_mode(dataset, 'VIP')

    train_df = pd.get_dummies(train_df, columns=['HomePlanet'])
    test_df = pd.get_dummies(test_df, columns=['HomePlanet'])
    combine = [train_df, test_df]
    train_df = pd.get_dummies(train_df, columns=['Destination'])
    test_df = pd.get_dummies(test_df, columns=['Destination'])
    combine = [train_df, test_df]

    for dataset in combine:
        le = LabelEncoder()
        dataset['VIP'] = le.fit_transform(dataset['VIP'])
    train_df = pd.get_dummies(train_df, columns=['CryoSleep'])
    test_df = pd.get_dummies(test_df, columns=['CryoSleep'])
    combine = [train_df, test_df]
    for dataset in combine:
        dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)  # разбиваем возраста на равные отрезки
    for dataset in combine:
        dataset.loc[dataset['Age'] <= 15, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 15) & (dataset['Age'] <= 31), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 31) & (dataset['Age'] <= 47), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 47) & (dataset['Age'] <= 63), 'Age'] = 3
        dataset.loc[(dataset['Age'] > 63), 'Age'] = 4
        dataset['Age'] = dataset['Age'].astype(int)

    for dataset in combine:
        replace_empty_cabin(dataset, 'Cabin')
    for dataset in combine:
        dataset['deck'] = dataset['Cabin'].apply(lambda x: str(x)[:1])
        dataset['num'] = dataset['Cabin'].apply(lambda x: x.split('/')[1])
        dataset['num'] = dataset['num'].astype(int)
        dataset['side'] = dataset['Cabin'].apply(lambda x: str(x)[-1:])
        dataset['deck'].fillna(dataset['deck'].mode()[0], inplace=True)
        dataset['num'].fillna(dataset['num'].mode()[0], inplace=True)
        dataset['side'].fillna(dataset['side'].mode()[0], inplace=True)
    deck_mapping = {"B": 1, "C": 1, "G": 2, "Z": 2, "A": 2, "F": 3, "D": 3, "E": 4, "T": 5}
    for dataset in combine:
        dataset['deck'] = dataset['deck'].map(deck_mapping)
    side_map = {'P': 1, 'S': 0}
    for dataset in combine:
        dataset['side'] = dataset['side'].map(side_map)
    for dataset in combine:
        dataset['side'].fillna(dataset['side'].mode()[0], inplace=True)
    for dataset in combine:
        dataset['region1'] = (dataset['num'] < 302.5).astype(int)
        dataset['region2'] = ((dataset['num'] >= 302.5) & (dataset['num'] < 600)).astype(int)
        dataset['region3'] = ((dataset['num'] >= 600) & (dataset['num'] < 900)).astype(int)
        dataset['region4'] = ((dataset['num'] >= 900) & (dataset['num'] < 1200)).astype(int)
        dataset['region5'] = ((dataset['num'] >= 1200) & (dataset['num'] < 1500)).astype(int)
        dataset['region6'] = ((dataset['num'] >= 1500) & (dataset['num'] < 1800)).astype(int)
        dataset['region7'] = (dataset['num'] > 1800).astype(int)
    for dataset in combine:
        dataset['group'] = dataset.PassengerId.apply(lambda x: x.split('_')[0])
        dataset['group'] = dataset['group'].astype(int)
    for dataset in combine:
        dataset['sum'] = dataset['VRDeck'] + dataset['Spa'] + dataset['ShoppingMall'] + dataset['RoomService'] + \
                         dataset['FoodCourt']
    for dataset in combine:
        dataset['vr'] = dataset['VRDeck'] / dataset['sum']
        dataset['spa'] = dataset['Spa'] / dataset['sum']
        dataset['room'] = dataset['RoomService'] / dataset['sum']
        dataset['shop'] = dataset['ShoppingMall'] / dataset['sum']
        dataset['food'] = dataset['FoodCourt'] / dataset['sum']

    # заполняем поля поделенные на ноль
    for dataset in combine:
        dataset['vr'].fillna(0, inplace=True)
        dataset['spa'].fillna(0, inplace=True)
        dataset['room'].fillna(0, inplace=True)
        dataset['shop'].fillna(0, inplace=True)
        dataset['food'].fillna(0, inplace=True)
    train_df = train_df.drop(['Name', 'PassengerId', 'AgeBin', 'Cabin', 'num'], axis=1)
    test_df = test_df.drop(['Name', 'PassengerId', 'AgeBin', 'Cabin', 'num'], axis=1)
    X_train = train_df.drop("Transported", axis=1)
    Y_train = train_df["Transported"]
    X_test = test_df.copy()
    X_test.to_csv("work_data/x_test.csv", index = False)
    X_train.to_csv("work_data/x_train.csv", index = False)
    Y_train.to_csv("work_data/y_train.csv", index = False)

class My_Classifier_Model:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None

    def train(self):
        format_data()

        model = CatBoostClassifier(iterations=300,
                                   learning_rate=0.15,
                                   depth=4,
                                   cat_features=[0],
                                   loss_function='MultiClass')
        x_train = pd.read_csv('work_data/x_train.csv')
        y_train = pd.read_csv('work_data/y_train.csv')
        model.fit(x_train, y_train)

        model_dir = './model'
        os.makedirs(model_dir, exist_ok=True)
        model_filepath = os.path.join(model_dir, 'model.pkl')
        with open(model_filepath, 'wb') as f:
            pickle.dump(model, f)

        self.logger.info('Model training completed.')

    def prediction(self):
        with open("model/model.pkl", 'rb') as f:
            model = pickle.load(f)
        preds_class = model.predict(pd.read_csv('work_data/x_test.csv'))
        preds_class = preds_class.T
        preds_class[0, :]
        test_df = pd.read_csv('data/sample_submission.csv')
        submission = pd.DataFrame({
            "PassengerId": test_df["PassengerId"],
            "Transported": preds_class[0, :]
        })
        submission.to_csv('data/submission.csv', index=False)
        self.logger.info('Prediction saved to submission.scv')

def main():
    parser = argparse.ArgumentParser(description='Model training and prediction')
    parser.add_argument('mode', choices=['train', 'predict'], help='Mode: train or predict')
    args = parser.parse_args()

    model = My_Classifier_Model()

    if args.mode == 'train':
        model.train()
    elif args.mode == 'predict':
        model.prediction()

if __name__ == '__main__':
    main()