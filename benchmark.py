import pandas as pd
import numpy as np
import tensorflow as tf
import tqdm

import time

import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import sklearn.model_selection


calibration_ds = pd.read_hdf('out.hdf','df')

calibration_ds['logCH4'] = np.log(calibration_ds['CH4 / ppm'])

V1Average = np.average(calibration_ds['V1C'])
V2Average = np.average(calibration_ds['V1C'])
V3Average = np.average(calibration_ds['V3C'])

V1Std = np.std(calibration_ds['V1C'])
V2Std = np.std(calibration_ds['V2C'])
V3Std = np.std(calibration_ds['V3C'])

logppm_mean = np.average(calibration_ds['logCH4'])
logppm_std = 10

calibration_ds['V1CNorm'] = (calibration_ds['V1C'] - V1Average)/V1Std
calibration_ds['V2CNorm'] = (calibration_ds['V2C'] - V2Average)/V2Std
calibration_ds['V3CNorm'] = (calibration_ds['V3C'] - V3Average)/V3Std

calibration_ds['logCH4Norm'] = (calibration_ds['logCH4'] - logppm_mean)/logppm_std
signals = np.column_stack((calibration_ds['V1CNorm'], calibration_ds['V2CNorm'], calibration_ds['V3CNorm']))

num_datapoints = np.shape(signals)[0]

print("Dataset Size: %d dps"%num_datapoints)

print("====== EVALUATING LINEAR REGRESSION MODEL =======")


lr_model_train_time = []
lr_model_inference_time = []

for i in tqdm.tqdm(range(0,50)):

    reg_model = sklearn.linear_model.LinearRegression()
    t0_train = time.perf_counter()
    reg_model.fit(signals, calibration_ds['logCH4Norm'].values)
    t1_train = time.perf_counter()
    
    tdelta_train = (t1_train - t0_train)

    lr_model_train_time.append(tdelta_train)
    
    t0_inference = time.perf_counter()
    reg_model.predict(signals)
    t1_inference = time.perf_counter()

    tdelta_inference = (t1_inference - t0_inference)

    lr_model_inference_time.append(tdelta_inference)
    
                                   
print("Lin Reg Training Time: %4.3e"%np.average(lr_model_train_time))
print("Lin Reg Inference Time (per signal): %4.3e"%(np.average(lr_model_inference_time)/num_datapoints))



print("====== EVALUATING QUADRATIC MODEL =======")

quadratic_model_train_time = []
quadratic_model_inference_time = []

for i in tqdm.tqdm(range(0,50)):

    poly2reg =make_pipeline(PolynomialFeatures(2),sklearn.linear_model.LinearRegression())
    t0_train = time.perf_counter()
    poly2reg.fit(signals, calibration_ds['logCH4Norm'].values)
    t1_train = time.perf_counter()
    
    tdelta_train = (t1_train - t0_train)

    quadratic_model_train_time.append(tdelta_train)
    
    t0_inference = time.perf_counter()
    poly2reg.predict(signals)
    t1_inference = time.perf_counter()

    tdelta_inference = (t1_inference - t0_inference)

    quadratic_model_inference_time.append(tdelta_inference)
    
                                   
print("Quadratic Training Time: %4.3e"%np.average(quadratic_model_train_time))
print("Quadratic Inference Time (per signal): %4.3e"%(np.average(quadratic_model_inference_time)/num_datapoints))


print("====== EVALUATING RANDOM FOREST MODEL =======")

rf_model_train_time = []
rf_model_inference_time = []

for i in tqdm.tqdm(range(0,50)):
    
    rf_model = sklearn.ensemble.RandomForestRegressor()
    t0_train = time.perf_counter()
    rf_model.fit(signals, calibration_ds['logCH4Norm'].values)
    t1_train = time.perf_counter()
    
    tdelta_train = (t1_train - t0_train)

    rf_model_train_time.append(tdelta_train)
    
    t0_inference = time.perf_counter()
    rf_model.predict(signals)
    t1_inference = time.perf_counter()

    tdelta_inference = (t1_inference - t0_inference)

    rf_model_inference_time.append(tdelta_inference)
    
                                   
print("Random Forest Training Time: %4.3e"%np.average(rf_model_train_time))
print("Random Forest Inference Time (per signal): %4.3e"%(np.average(rf_model_inference_time)/num_datapoints))




print("====== EVALUATING SVM MODEL =======")

svm_model_train_time = []
svm_model_inference_time = []

for i in tqdm.tqdm(range(0,50)):
    
    svr_model = sklearn.svm.NuSVR()
    t0_train = time.perf_counter()
    svr_model.fit(signals, calibration_ds['logCH4Norm'].values)
    t1_train = time.perf_counter()
    
    tdelta_train = (t1_train - t0_train)

    svm_model_train_time.append(tdelta_train)
    
    t0_inference = time.perf_counter()
    svr_model.predict(signals)
    t1_inference = time.perf_counter()

    tdelta_inference = (t1_inference - t0_inference)

    svm_model_inference_time.append(tdelta_inference)
    
                                   
print("SVM Training Time: %4.3e"%np.average(svm_model_train_time))
print("SVM Inference Time (per signal): %4.3e"%(np.average(svm_model_inference_time)/num_datapoints))

print("====== EVALUATING ANN / 15 HL / TanH MODEL =======")

epochs = 100000
patience = 100
hl_size = 15
af = 'tanh'


ann_model_train_time = []
ann_model_inference_time = []
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
    
for i in tqdm.tqdm(range(0,10)):
    
    
    ann_model = tf.keras.Sequential()
    ann_model.add(tf.keras.Input(shape=signals.shape[1]))
    ann_model.add(tf.keras.layers.Dense(hl_size, activation=af))
    ann_model.add(tf.keras.layers.Dense(1, activation=af))
    
    ann_model.compile(optimizer='adam',
                                  loss=tf.keras.losses.MeanSquaredError(),
                                  metrics=[])
    
    t0_train = time.perf_counter()
    ann_model.fit(x=signals, y=calibration_ds['logCH4Norm'].values, epochs=epochs, verbose=0, callbacks=[early_stop], shuffle=True)     
    t1_train = time.perf_counter()
    
    tdelta_train = (t1_train - t0_train)

    ann_model_train_time.append(tdelta_train)
    
    t0_inference = time.perf_counter()
    ann_model.predict(signals)
    t1_inference = time.perf_counter()

    tdelta_inference = (t1_inference - t0_inference)

    ann_model_inference_time.append(tdelta_inference)
    

print("ANN Training Time: %4.3e"%np.average(ann_model_train_time))
print("ANN Inference Time (per signal): %4.3e"%(np.average(ann_model_inference_time)/num_datapoints))