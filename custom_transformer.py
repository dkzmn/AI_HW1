import numpy as np
from sklearn.base import TransformerMixin


class MyTransformer(TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['mileage'] = X['mileage'].str.split(' ').str[0]
        X['engine'] = X['engine'].str.split(' ').str[0]
        X['max_power'] = X['max_power'].str.split(' ').str[0]
        X[['mileage', 'engine', 'max_power']] = X[['mileage',
                                                   'engine', 'max_power']].replace('', np.nan).astype(float)

        X['torque'] = X['torque'].str.lower().str.replace(
            'at', '@').str.replace('/', '@')
        X['max_torque_rpm'] = X['torque'].str.split(
            '@').str[1].astype(str).str.replace(',', '').str.extract(r'(\d*-?\d+)')
        X['max_torque_rpm'] = X['max_torque_rpm'].str.split(
            '-').str[-1].astype('float')
        X['torque_new'] = X['torque'].str.split(
            '@').str[0].str.extract(r'(\d*\.?\d+)').astype('float')
        X['torque_new'] = X['torque_new'].where(
            X['torque'].str.contains('kgm') == False, X['torque_new'] * 9.81)
        X.drop('torque', axis=1, inplace=True, errors='ignore')
        X.rename(columns={"torque_new": "torque"}, inplace=True)

        X['mileage'] = X['mileage'].fillna(X['mileage'].median())
        X['engine'] = X['engine'].fillna(X['engine'].median())
        X['max_power'] = X['max_power'].fillna(X['max_power'].median())
        X['seats'] = X['seats'].fillna(X['seats'].median())
        X['torque'] = X['torque'].fillna(X['torque'].median())
        X['max_torque_rpm'] = X['max_torque_rpm'].fillna(
            X['max_power'] * 0.7457 * 9550 / X['torque'])

        X[['engine', 'seats']] = X[['engine', 'seats']].astype(int)
        X[['engine', 'seats']] = X[['engine', 'seats']].astype(int)
        X['name'] = X['name'].str.split(
            ' ').str[0] + ' ' + X['name'].str.split(' ').str[1]
        return X
