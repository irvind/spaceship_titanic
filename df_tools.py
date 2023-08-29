import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def augument_cabin(df):
    deck_values = []
    num_values = []
    side_values = []
    for val in df['Cabin']:
        if isinstance(val, str):
            tokens = val.split('/')
            assert len(tokens) == 3
            deck_values.append(tokens[0])
            num_values.append(int(tokens[1]))
            side_values.append(tokens[2])
        else:
            deck_values.append(np.nan)
            num_values.append(np.nan)
            side_values.append(np.nan)

    df['Deck'] = pd.Series(deck_values, dtype='object')
    df['CabinIndex'] = pd.Series(num_values, dtype='object')
    df['Side'] = pd.Series(side_values, dtype='object')

    
def impute_general_cols(df):
    cat_and_bool_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    values_before_imputer = df[['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']].values
    values_after_imputer = cat_and_bool_imputer.fit_transform(values_before_imputer)

    df['HomePlanet'] = pd.Series(values_after_imputer[:, 0], dtype='category')
    df['CryoSleep'] = pd.Series(values_after_imputer[:, 1], dtype=bool)
    df['Destination'] = pd.Series(values_after_imputer[:, 2], dtype='category')
    df['VIP'] = pd.Series(values_after_imputer[:, 3], dtype=bool)
    df['Deck'] = pd.Series(values_after_imputer[:, 4], dtype='category')
    df['Side'] = pd.Series(values_after_imputer[:, 5], dtype='category')

    expenses_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    expenses_imputer = SimpleImputer(missing_values=np.nan, strategy='constant')
    values_before_imputer = df[expenses_columns].values
    values_after_imputer = expenses_imputer.fit_transform(values_before_imputer)

    for idx, col in enumerate(expenses_columns):
        df[col] = pd.Series(values_after_imputer[:, idx], dtype='float64')

    age_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    values_before_imputer = df[['Age']].values
    values_after_imputer = age_imputer.fit_transform(values_before_imputer)
    df['Age'] = pd.Series(values_after_imputer[:, 0], dtype='float64')

    
def impute_name(df):
    name_values = df['Name'].values
    passenger_id_values = df['PassengerId'].values
    name_null_indicies = np.argwhere(df['Name'].isnull().values).flatten()
    name_values[name_null_indicies] = 'UnknownName_' + passenger_id_values[name_null_indicies]
    df['Name'] = pd.Series(name_values, dtype='object')
    
    
def impute_cabin_index(df):
    cabin_index_values = df['CabinIndex'].values
    cabin_index_null_indicies = np.argwhere(df['CabinIndex'].isnull().values).flatten()
    cabin_index_values[cabin_index_null_indicies] = list(range(len(cabin_index_null_indicies)))
    df['CabinIndex'] = pd.Series(cabin_index_values, dtype='int32')


def fill_cabin(df):
    cabin_values = df['Cabin'].values
    cabin_null_indicies = np.argwhere(df['Cabin'].isnull().values).flatten()
    for idx in cabin_null_indicies:
        item = df.iloc[idx]
        cabin_values[idx] = '{}/{}/{}'.format(item['Deck'], str(item['CabinIndex']), item['Side'])
    df['Cabin'] = pd.Series(cabin_values, dtype='object')


def augument_group_size(df):
    col_data = [val.split('_') for val in df['PassengerId']]
    counter = {}
    for group_id, group_pos in col_data:
        if group_id not in counter:
            counter[group_id] = 0
        counter[group_id] += 1

    group_size = []
    for group_id, group_pos in col_data:
        group_size.append(counter[group_id])
    df['GroupSize'] = pd.Series(group_size, dtype='int32')


def transform_price_columns(df):
    cols = ['ShoppingMall', 'RoomService', 'FoodCourt', 'Spa', 'VRDeck']
    for col in cols:
        df[col] = pd.Series(np.log1p(df[col].values))


def dataset_to_np_array(df, with_y=True):
    df2 = df[['CryoSleep', 'Age', 'VIP', 'RoomService', 'GroupSize']]
    for col in ('CryoSleep', 'VIP'):
        df2[col] = df2[col].astype(int)

    X_cat = df[['HomePlanet', 'Destination', 'Deck', 'Side']].values
    one_hot_encoder = OneHotEncoder(categories='auto', drop='first')
    col_transformer = ColumnTransformer([
        ('HomePlanet_onehot', one_hot_encoder, [0]),
        ('Destination_onehot', one_hot_encoder, [1]),
        ('Deck_onehot', one_hot_encoder, [2]),
        ('Side_onehot', one_hot_encoder, [3]),
    ])
    X_cat_transformed = col_transformer.fit_transform(X_cat).toarray()
    # print(X_cat_transformed.shape)
    # X_cat_transformed
    # for i in range(4):
    #     print(col_transformer.transformers_[i][1].categories_)

    X_regular = df2.values
    X = np.hstack((X_regular, X_cat_transformed))
    # print(X.shape)
    if with_y:
        y = df['Transported'].astype(int).values
        return X, y
    else:
        return X


def prepare_dataset(df, with_y=True):
    augument_cabin(df)
    impute_general_cols(df)
    impute_name(df)
    impute_cabin_index(df)
    fill_cabin(df)
    augument_group_size(df)
    transform_price_columns(df)

    return dataset_to_np_array(df, with_y)
