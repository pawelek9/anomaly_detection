
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import time


class DataPrepare():

    def __init__(self):
        self.path =  r'C:\Users\pawelkl\Desktop\test_py\licencjat\input.csv'
        self.dataframe = pd.read_csv(self.path, sep= ',', encoding= 'latin-1')

    def create_data_info(self):
        print('min date: ', self.dataframe.Data.min(),'max date: ', self.dataframe.Data.max())
        print('table shape: ', self.dataframe.shape)
        print('column names')
        print(self.dataframe.columns[:10].values)

    def create_data(self):
        self.dataframe.replace('bd', np.nan, inplace= True)
        self.dataframe.replace(',', '.', inplace=True)
        self.dataframe['Data'] = pd.to_datetime(self.dataframe['Data'])
        self.dataframe.set_index('Data', inplace= True)


class Models():

    def __init__(self, data):
        self.data = data

    def prepare_date(self, col):
        data = self.data[col].dropna()
        # shape = data.shape[0]
        # t = range(1, shape+1)
        data = pd.DataFrame(data)
        # print(data.shape, len(t))
        # data['t'] = t
        return data

    def create_models(self):
        data = self.data
        start_time = time.time()
        results = {'kmeans_results': pd.concat({col: self.kmeans_v_2(col) for col in data.columns}),
                   'lof_models': pd.concat({col: self.lof_model(col, p=2) for col in data.columns}),
                   'if_results': pd.concat({col: self.if_model_v2(col, n_estimators=100) for col in data.columns})}
        self.results = pd.concat(results)
        self.results.reset_index(inplace=True)
        self.results.drop(columns='level_2', inplace= True)
        self.results.columns = ['algorithms', 'funds_name', 'Data', 'value',
                                'cluster', 'cluster_center', 'diff', 'mean_diff',
                                'std_diff', 'critical_value', 'outlier', 't', 'Series']
        print((time.time() - start_time)/60, 'execution time')

    def kmeans_v_2(self, col):
        x = self.prepare_date(col)
        shape = self.data.shape[0]
        target_mapper = x.shape[0] / shape
        k = np.where(target_mapper < 0.1, 2,
                     np.where(target_mapper < 0.2, 4,
                              np.where(target_mapper < 0.5, 6,
                                       np.where(target_mapper < 0.6, 10,
                                                np.where(target_mapper < 0.7, 13, 17)))))
        kmeans = KMeans(n_clusters=int(k))
        x = x.iloc[:, :1]
        print(x.shape)
        kmeans.fit(x)
        x['cluster'] = kmeans.predict(x)
        x.reset_index()
        cluster_centers = pd.DataFrame(kmeans.cluster_centers_.reshape(-1))
        cluster_centers['label'] = cluster_centers.index.values
        cluster_centers.columns = ['cluster_center', 'cluster']

        cluster_centers.columns = ['cluster_center', 'cluster']
        x.reset_index(inplace=True)
        x = x.merge(cluster_centers, on='cluster')
        x['diff'] = np.abs(x[x.columns[1]] - x[x.columns[-1]])
        x = x.merge(
            pd.DataFrame(
                {'mean_diff': x.groupby('cluster').mean()['diff'],
                 'std_diff': x.groupby('cluster').std()['diff']}
            ).reset_index(), on='cluster'
        )
        ##here will be 3 ways to deal with:
        ##one that comuptes critical value adding mean and std
        ##multipylin mean by constans
        ##there adding mean and std then multipylin by constans
        x['critical_value'] = x['mean_diff'] + 1.2 * x['std_diff']
        x['outlier'] = np.where(x['diff'] > x['critical_value'], True, False)
        x.sort_values('Data', inplace=True)
        x['t'] = np.array(range(x.shape[0])) + 1
        x.rename(columns={col: 'value'}, inplace=True)
        x['Series'] = col
        print('k means', k)

        return x

    def lof_model(self, col, p):
        x = self.prepare_date(col)
        shape = self.data.shape[0]
        target_mapper = x.shape[0] / shape
        n = np.where(target_mapper < 0.1, 3,
                     np.where(target_mapper < 0.2, 7,
                                                np.where(target_mapper < 0.7, 10, 12)))

        lof = LocalOutlierFactor(n_neighbors=int(n), n_jobs=-1, p=p)
        x = x.iloc[:, :1]
        print(x.shape)
        print('lof', n)
        lof.fit(x)
        predicted = lof.fit_predict(x)

        x['outlier'] = np.where(predicted == -1, True, False)
        x.reset_index(inplace=True)
        x['t'] = x.reset_index().iloc[:, 0]
        x.rename(columns={col: 'value'}, inplace=True)
        x['Series'] = col

        return x

    def if_model_v2(self, col, n_estimators):
        x = self.prepare_date(col)
        shape = self.data.shape[0]
        target_mapper = x.shape[0] / shape
        cont = np.where(target_mapper < 0.1, 0.1,
                            np.where(target_mapper < 0.5, 0.07,
                                    np.where(target_mapper < 0.7, 0.06, 0.05)))

        clf = IsolationForest(n_estimators=n_estimators,
                              n_jobs=-1,
                              contamination=cont,
                              max_samples=0.8)
        x = x.iloc[:, :1]
        x['t'] = np.array(range(x.shape[0])) + 1
        print(x.shape)

        clf.fit(x)
        predicted = clf.fit_predict(x)

        x['outlier'] = np.where(predicted == -1, True, False)
        x.reset_index(inplace=True)
        x['t'] = x.reset_index().iloc[:, 0]
        x.rename(columns={col: 'value'}, inplace=True)
        x['Series'] = col
        print('if', cont)

        return x


class Mapper():

    def __init__(self, data):
        self.data = data

    def create_mapper(self):
        self.df = pd.Series({col: 0 if len(self.data[col].dropna())< 300 else 1 for col in self.data.columns})
        self.df = pd.DataFrame(self.df)
        self.df['mapper'] = np.where(self.df  == 1, 'if_results', 'kmeans_results')
        self.df.reset_index(inplace=True)
        self.df.columns = ['funds_name', 'rq', 'algorithms']

 
