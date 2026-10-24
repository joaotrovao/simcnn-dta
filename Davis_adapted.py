#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bioinformática - Projeto Final

Ariana Lopes
João Trovão

"""
#%%
from tdc.multi_pred import DTI
from math import log
import numpy as np
dataDavis = DTI(name = 'DAVIS')
davis_Dict=dataDavis.get_data(format = 'dict')

def log_dav(x):
    return -log((x/1e9),10)
log_arr = np.vectorize(log_dav)
davis_Dict['Y']=log_arr(davis_Dict['Y'])


#%%

# K2: Smith-Waterman Matrix

from skbio import Protein
from skbio.alignment import local_pairwise_align_protein
from math import sqrt
import numpy as np


K2=np.zeros((5,5))
for j in range(5):
    s1 = Protein(davis_Dict['Target'][j])
    for i in range(5):
        if i>=j:
            s2 = Protein(davis_Dict['Target'][i])
            alignment, score, start_end_positions = local_pairwise_align_protein(s1, s2, 11, 1)
            alignment_s1, score_s1, start_end_positions_s1 = local_pairwise_align_protein(s1, s1, 11, 1)
            alignment_s2, score_s2, start_end_positions_s2 = local_pairwise_align_protein(s2, s2, 11, 1)
            score_normalized=score/sqrt(score_s1*score_s2)
            K2[i,j]=(score_normalized)
            K2[j,i]=(score_normalized)


#%%

# K1: Tanimoto Coefficient Matrix

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

K1=np.zeros((68,68))

def tanimoto_calc(smi1, smi2):
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 3, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 3, nBits=2048)
    s = round(DataStructs.TanimotoSimilarity(fp1,fp2),3)
    return s

for j in range(68):
    d1 = davis_Dict['Drug'][j*379]
    for i in range(68):
        if i>=j:
            d2 = davis_Dict['Drug'][i*379]
            tanimoto_coef=tanimoto_calc(d1,d2)
            K1[i,j]=(tanimoto_coef)
            K1[j,i]=(tanimoto_coef)


#%% 

# K1 * K2

Ktotal=np.zeros((340,68,5))
idx=0

for i in range(68):
    for j in range(5):
        K=np.zeros((68,5))
        for m in range(68):
            for n in range(5):
                K[m,n]=K1[i,m]*K2[j,n]
        Ktotal[idx]=K

        idx=idx+1


#%%
import numpy as np
Y=np.zeros(340)
idx=0
for j in range(68):
    for i in range(5):
        Y[idx]=davis_Dict['Y'][j*379+i]
        idx=idx+1


#%% dataset split
import numpy as np
indices = np.arange(340)
from sklearn.model_selection import train_test_split
train, test = train_test_split(indices, test_size=0.2, train_size=None) # 80/20 (train/test)

#%%

import tensorflow as tf
tf.autograph.set_verbosity(0)
tf.autograph.experimental.do_not_convert
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)


from tensorflow.keras.layers import Input, Conv2D, Dense, Concatenate, Embedding, Attention, LeakyReLU, Dropout,Flatten,MaxPool2D,Reshape,Permute
from tensorflow.keras.models import Model


inMatrix = Input(shape=(68,5,1)) 
x = Conv2D(filters=32,kernel_size=5,activation='relu')(inMatrix)

z=Flatten()(x)

z = Dropout(0.1)(z)
z = Dense(128, activation=LeakyReLU(alpha=0.01))(z)
z = Dense(1)(z)

model = Model(inputs=inMatrix, outputs=z)

layer_outputs = [layer.output for layer in model.layers]
viz_model = Model(inputs=inMatrix, outputs=layer_outputs)

model.summary()


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
opt = Adam(learning_rate=0.001)
model.compile(loss=MeanSquaredError(), optimizer=opt)

print("[INFO] training model...")
history=model.fit(x=Ktotal, y=Y, epochs=100, batch_size=50)

scores = model.evaluate(Ktotal[test], Y[test], verbose=0)

import numpy as np

print("[INFO] predicting binding affinities...")
preds = model.predict(Ktotal[test])

y_true=np.squeeze(np.asarray(Y[test]))
y_pred=np.squeeze(np.asarray(preds))

#mean squared error
from tensorflow.keras.metrics import mean_squared_error,MeanSquaredError
mse=mean_squared_error(y_true,y_pred)

#concordance index
import pandas as pd
from lifelines.utils import concordance_index
dfCI = pd.DataFrame(data={'y_true': y_true, 'y_pred': y_pred})
ci=concordance_index(y_true, y_pred)

#AUPRC
# convertion to binary classes
y_true_bin=np.where(y_true>=7, 1, y_true)
y_true_bin=np.where(y_true<7, 0, y_true_bin).astype(int)

y_pred_bin=np.where(y_pred>=7, 1, y_pred)
y_pred_bin=np.where(y_pred<7, 0, y_pred_bin).astype(int)

import sklearn.metrics
auprc = sklearn.metrics.average_precision_score(y_true_bin, y_pred_bin)

#%% plot loss
import matplotlib.pyplot as plt
import numpy
plt.figure()
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend('train', loc='upper left')
plt.show()

#%% print metrics
print('------------------------------------------------------------------------')
print('Scores:')
print(f'> MSE: {mse} )')
print(f'> CI: {ci} )')
print(f'> AUPRC: {auprc} )')
print('------------------------------------------------------------------------')

#%% predicted vs true binding affinities plot
plt.figure()
plt.plot(y_true,y_pred,'o', color='black')
plt.plot(np.arange(5,20),np.arange(5,20),'--', color='black')
plt.title('True vs Predicted')
plt.ylabel('Predicted')
plt.xlabel('True')
plt.show()
