#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 10:28:17 2023

@author: anisio
"""
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()
import numpy as np
from math import sqrt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras.optimizers import Adam

from tensorflow import keras
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.python.framework.ops import disable_eager_execution
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
disable_eager_execution()


import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import csv
import datetime
import time

# field names 
fields = ['Method', 'Location', 'Folds_Mean', 'Folds_STD', 'RMSE_Predicted','Scoring_Metrics','TE_MEAN','TE_SUM'] 


## Dados de estudo ##
save_path = "/home/anisio/Documentos/Doutorado/Tese/Pesos"
path = "/home/anisio/Documentos/Doutorado/Tese/Dados"
dadosCampinaVerde = pd.read_csv(path + '/A519.csv',sep=';')
dadosSorriso = pd.read_csv(path + '/A904.csv',sep=';')
dadosDiamante = pd.read_csv(path + '/A849.csv',sep=';')
dadosCampo = pd.read_csv(path + '/A884.csv',sep=';')

listaDados = [dadosCampo,dadosSorriso,dadosDiamante,dadosCampinaVerde]
listNames = ["RS-CAMPOBOM","MT-SORRISO","PR-DIAMANTE DO NORTE","MG-CAMPINA VERDE"]
listLegends = ['1 - LR','2 - LASSO','3 - EN','4 - KNN','5 - CART', '6 - SVR', '7 - PINN', '8 - MPL-Convencional']

test_size= 0.2
learning_rate=1e-3
num_folds = 8
epochs = 3000
seed = 7

now = datetime.datetime.now()
formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
print(now)

file_path = "/home/anisio/Documentos/Doutorado/Tese/analise/analise_others_1_"+formatted_datetime+ " .csv"


listDadosLimpos =[]

for df in listaDados:
    df = df[['data','hora', 'umid_inst', 'pto_orvalho_inst', 'pressao','radiacao', 'vento_vel','temp_inst']]
    
    
    # Converter a coluna 'hora' para string
    df['hora'] = df['hora'].astype(str)
    df['hora_'] = df['hora'].apply(lambda x: x.zfill(4))
    
    
    # Juntar os campos 'data' e 'hora' em um único campo 'data_hora' do tipo datetime
    #df['data_'] = # Juntar os campos 'data' e 'hora' em um único campo 'data_hora' do tipo datetime
    df['data_hora'] = pd.to_datetime(df['data'] + '-' + df['hora_'], format='%Y-%m-%d-%H%M')
    
    # Converter a coluna 'hora' para o tipo 'int'
    df['hora'] = df['hora'].astype(int)
    
    # Converter as colunas restantes para o tipo 'float'
    colunas_float = ['pressao', 'radiacao', 'temp_inst', 'umid_inst', 'vento_vel']
    df[colunas_float] = df[colunas_float].astype(float)
    
    df = df.dropna()
    listDadosLimpos.append(df)

df1 = listDadosLimpos[0]
for i in range(3):
    listPareada = []
    for df in listDadosLimpos:
        # Mesclar os dataframes com base nas colunas 'data' e 'hora'
        df_merged = df.loc[df['data_hora'].isin(df1['data_hora'])]
        df1 = df_merged
        
        # Selecionar as datas em df2 que existem em df1
        #df = df[df['data'].isin(df1['data'])]
        listPareada.append(df_merged[['umid_inst', 'pto_orvalho_inst', 'pressao','radiacao', 'vento_vel','temp_inst']])
r =[] 
for i in range(4):
    df_p = pd.DataFrame(listPareada[i])
    r.append(df_p .describe())
        
cont = 0
p_values =[]
# Test options and evaluation metric
scoring = 'neg_root_mean_squared_error'
total_results = []


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('.', end=' ')
        if epoch % 100 == 0:
            print(datetime.datetime.now())
            print(f"Epoch {epoch+1}: loss = {logs['loss']} rmse= {logs['root_mean_squared_error']}")
        if epoch + 1 % epochs == 0:
            self.model.save_weights(f"{save_path}/model_weights_epoch{epoch+1}.h5")
        
with open(file_path, 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(fields)
result_file = []
for df in listPareada:
    
    print(listNames[cont])
    
    # Preprocess data
    df = pd.DataFrame(df).dropna()
    X = df[['umid_inst', 'pto_orvalho_inst', 'pressao','radiacao', 'vento_vel']].to_numpy()
    y = df['temp_inst'].to_numpy()
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    

    
    # Standardize the dataset
    pipelines = []
    pipelines.append(('1', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
    pipelines.append(('2', Pipeline([('Scaler', StandardScaler()),('LASSO',Lasso())])))
    pipelines.append(('3', Pipeline([('Scaler', StandardScaler()),('EN',ElasticNet())])))
    pipelines.append(('4', Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsRegressor())])))
    pipelines.append(('5', Pipeline([('Scaler', StandardScaler()),('CART',DecisionTreeRegressor())])))
    pipelines.append(('6', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))

    results = []
    resultsBar = []
    names = []
    
    for name, modelo in pipelines:
        inicio = time.time()
        
        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
        cv_results = abs(cross_val_score(modelo, X_train, y_train, cv=kfold, scoring=scoring))
        fim = time.time()
        
        results.append(cv_results)
        resultsBar.append(cv_results.mean())
        names.append(name)
        
        modelo.fit( X_train, y_train)
        
        previsto = modelo.score( X_train, y_train)
        
        tempo_execucao = fim - inicio
        y_prev = modelo.predict(X_test)
        
        result_file.append([name,listNames[cont],cv_results.mean(),cv_results.std(),previsto,cv_results,scoring,tempo_execucao,0])
        
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    cont += 1
    
values = ','.join(str(v) for v in result_file)
with open(file_path, 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(result_file)

    print("List saved to CSV file successfully.")
        
    
    
    
    
    
    
    
    callback = MyCallback()
    tf.keras.backend.clear_session()
    
    inputs_a = keras.layers.Input(shape=(X_train.shape[1],))
    hidden1_a = keras.layers.Dense(320, activation="relu")(inputs_a)
    hidden2_a = keras.layers.Dense(80, activation="relu")(hidden1_a)
    hidden3_a = keras.layers.Dense(80, activation="relu")(hidden2_a)
    hidden4_a = keras.layers.Dense(80, activation="relu")(hidden3_a)
    hidden5_a = keras.layers.Dense(16, activation="relu")(hidden4_a)
    hidden6_a = keras.layers.Dense(12, activation="relu")(hidden5_a)
    hidden7_a = keras.layers.Dense(7, activation="relu")(hidden6_a)
    output_a = keras.layers.Dense(1)(hidden7_a)
    model_a = keras.models.Model(inputs=inputs_a, outputs=output_a)
        
    
   
    inputs = keras.layers.Input(shape=(X_train.shape[1],))
    hidden1 = keras.layers.Dense(320, activation="relu")(inputs)
    hidden2 = keras.layers.Dense(80, activation="relu")(hidden1)
    hidden3 = keras.layers.Dense(80, activation="relu")(hidden2)
    hidden4 = keras.layers.Dense(80, activation="relu")(hidden3)
    hidden5 = keras.layers.Dense(16, activation="relu")(hidden4)
    hidden6 = keras.layers.Dense(12, activation="relu")(hidden5)
    hidden7 = keras.layers.Dense(7, activation="relu")(hidden6)
    output = keras.layers.Dense(1)(hidden7)
    model = keras.models.Model(inputs=inputs, outputs=output)
    
   
    vet =[]
    def heat_eq_loss(y_true, y_pred):
       # Define heat equation parameters
       k = 0.024  # Thermal conductivity of air (W/mK)
       rho = 1.225  # Density of air (kg/m³)
       Cp = 1005  # Specific heat capacity of air (J/kgK)
       alpha = k / (rho * Cp)  # Thermal diffusivity of air (m²/s)

       # Calculate second derivative of predicted temperature
       d2T_dx2 = tf.gradients(tf.gradients(y_pred, inputs)[0], inputs)[0]
       d2T_dx2 = tf.where(tf.math.is_finite(d2T_dx2), d2T_dx2, tf.zeros_like(d2T_dx2))
       
       # Calculate residual of heat equation
       residual = tf.math.reduce_mean((d2T_dx2 * alpha)) #testar sem o pred e multiplicado
       #return residual
       vet.append(residual)
       
       return tf.math.reduce_mean(tf.square(y_true - y_pred)) - residual
    
    model.compile(loss=heat_eq_loss, metrics=[RootMeanSquaredError()], optimizer=Adam(learning_rate = learning_rate), experimental_run_tf_function=False)
    model_a.compile(loss=tf.keras.losses.MeanSquaredError(), metrics=[RootMeanSquaredError()], optimizer=Adam(learning_rate = learning_rate), experimental_run_tf_function=False)

    
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    pinn = []
    dl = []
    rmse = []
    rmse_dl = []
    pinn_time=[]
    fold_enumerate = 1
    for i, (train_index, test_index) in enumerate(kfold.split(X_train)):
        X_fold_train = X_train[train_index]
        y_fold_train = y_train[train_index]
        X_fold_test = X_train[test_index]
        y_fold_test = y_train[test_index]
        
        inicio = time.time()
        print("PINN({}) City({}) Time:({})".format(fold_enumerate,listNames[cont],inicio))
        history = model.fit(X_fold_train, y_fold_train, epochs=epochs, batch_size=16, validation_data=(X_fold_test, y_fold_test), verbose=0,callbacks=[callback])
        fim = time.time()
        
        pinn_time.append(fim - inicio)
        
        rmse.append(model.evaluate(X_fold_test, y_fold_test)[1])
        y_pred = model.predict(X_fold_test)
        pinn.append(sqrt(mean_squared_error(y_fold_test, y_pred)))
        fold_enumerate = fold_enumerate + 1
    y_prev_pinn = model.predict(X_test)
    
    
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    pinn = []
    dl = []
    rmse = []
    rmse_dl = []
    mlp_time=[]
    fold_enumerate = 1
    for i, (train_index, test_index) in enumerate(kfold.split(X_train)):
        X_fold_train = X_train[train_index]
        y_fold_train = y_train[train_index]
        X_fold_test = X_train[test_index]
        y_fold_test = y_train[test_index]
        
        inicio = time.time()
        print(f"MLP({fold_enumerate}) City({listNames[cont]}) Time:({inicio})")
        history_1 = model_a.fit(X_fold_train, y_fold_train, epochs=epochs, batch_size=16, validation_data=(X_fold_test, y_fold_test), verbose=0,callbacks=[callback])
        fim = time.time()
        
        mlp_time.append(fim - inicio)
        
        rmse_dl.append(model.evaluate(X_fold_test, y_fold_test)[1])
        y_pred = model.predict(X_fold_test)
        dl.append(sqrt(mean_squared_error(y_fold_test, y_pred)))
        fold_enumerate = fold_enumerate + 1
    y_prev_mlp = model_a.predict(X_test)
    
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title("Loss_PINN")
    plt.legend(loc='best')
    plt.show()
    
    plt.plot(history_1.history['loss'], label='loss')
    plt.plot(history_1.history['val_loss'], label='val_loss')
    plt.title("Loss_DL")
    plt.legend(loc='best')
    plt.show()
    
    
    names.append('7')
    rmse = np.array(rmse)
    pinn_array = np.array(pinn)
    resultsBar.append(pinn_array.mean())
    results.append(pinn_array)
    result_file.append(['7',listNames[cont],pinn_array.mean(),pinn_array.std(),rmse.mean(), pinn,"neg_root_mean_squared_error",np.array(pinn_time).sum(),np.array(pinn_time).mean(),np.array(y_prev_pinn)])
    
    
    names.append('8')
    rmse_dl = np.array(rmse_dl)
    dl_array = np.array(dl)
    resultsBar.append(dl_array.mean())
    results.append(dl_array)
    result_file.append(['8',listNames[cont],dl_array.mean(),dl_array.std(),rmse_dl.mean(), dl,"neg_root_mean_squared_error",np.array(mlp_time).sum(),np.array(mlp_time).mean(),np.array(y_prev_mlp)])
    
    values = ','.join(str(v) for v in result_file)
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result_file)

    print("List saved to CSV file successfully.")
 
    
   
    # Compare Algorithms
    fig = plt.figure()
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle("%s: (%s)" % ('Algorithm Comparison', listNames[cont]))
    ax = fig.add_subplot(111)
    boxplot = plt.boxplot(results)
    plt.xticks(range(1, len(names) + 1), names)
    
    legend_labels = listLegends
    plt.legend(boxplot['boxes'], legend_labels, loc='upper right')

    plt.show()
        
    
    fig = plt.figure()
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle("%s: (%s)" % ('Algorithm Comparison', listNames[cont]))
    ax = fig.add_subplot(111)
    barplot = plt.bar(range(len(names)), resultsBar, align='center')
    
    # Set x-axis tick labels
    plt.xticks(range(len(names)), names)
    
    # Add legend
    legend_labels = listLegends
    handles = barplot[:len(legend_labels)]  # Subset of barplot matching the legend_labels length
    
    plt.legend(handles, legend_labels, loc='upper right')
    
# =============================================================================
#     ax.yaxis.set_major_locator(MultipleLocator(0.2))
#     plt.yticks(range(0, 5, 1)) 
# =============================================================================
    plt.show()
    
    total_results.append(results)
    cont += 1






































        
        

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
