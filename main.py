from backend.functions import *

DataPrepare = DataPrepare()

#Printing informations about data
DataPrepare.create_data_info()
DataPrepare.create_data()

print(DataPrepare.dataframe.dtypes)

models = Models(DataPrepare.dataframe)
models.create_models()

mapper = Mapper(DataPrepare.dataframe)
mapper.create_mapper()
results = mapper.df.merge(models.results, how= 'left', on=['funds_name', 'algorithms'])

results.to_csv(r'D:\Desktop\APLIKACJE\studia\licencjat\projekt\results.csv')
