# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:10:25 2022

@author: Marcin
"""

import pandas as pd
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.fft import fft

print(df_info['class'].value_counts())


print(df_info[df_info['fold']==10]['class'].value_counts())

filename = r"D:\Project_sounds\fold1\7061-6-0-0.wav"

# Ładowanie pliku
y, sr = librosa.load(filename, sr=44100)
# Wizualizacja
plt.figure(figsize=(14,5))
plt.suptitle('Postać czasowa')
librosa.display.waveshow(y, sr = sr)

# Ekstrakcja cech

# 1. Zero crossing rate
zcr = librosa.zero_crossings(y+0.01)
changeSign = np.sum(zcr)

# 2. Energia
hop_length = 1
frame_length = 1

energy = np.array([
    sum(abs(y[i:i+frame_length]**2))
    for i in range(0, len(y), hop_length)
])


plt.plot(energy, 'ro')

energyValue = sum(energy) / len(energy)


# 3. RMSE
rmseValue = math.sqrt(np.mean(y*y))


# 4. FFT
FFT = fft(y)
FFT_mag = np.absolute(FFT)
f = np.linspace(0, sr, len(FFT_mag))

plt.plot(figsize=(14,5))
plt.plot(f, FFT_mag)
plt.xlabel('Częstotliwosc [Hz]')

# Zoom
plt.plot(f[:4000], FFT_mag[:4000])

# Spectral centroid
cent = librosa.feature.spectral_centroid(y=y,sr=sr)
plt.semilogy(cent.T, label='Spectral centroid')

# Spectral roll off
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
plt.plot(rolloff.T, label = 'Roll-off freq')

# Wywietlmy spektrogram 
X = librosa.stft(y)
Xdb = librosa.amplitude_to_db(abs(X))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()

# MFCC
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
librosa.display.specshow(mfccs, x_axis='time', sr=sr)
plt.colorbar()
plt.title('MFCCs')


# Ekstrakcja dla katalogów fold 1 - 10

# Metadata
df_info = pd.read_csv("UrbanSound8K.csv")


dict_list = []

# Iteracja po wszystkich foldach (katalogi)
for fold in range(1, 11):
    # Iteracja po plikach wewnątrz katalogów (fold)
    for filename in os.listdir('fold'+str(fold)):
        # Wczytaj dźwięk z librosa
        y, sr = librosa.load(('fold'+str(fold)+'\\'+filename), sr = 44100, mono = True)

        # Słownik na dane
        data_dict = {}
        
        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y+0.01)
        zcrAvg = np.sum(zcr) / zcr.shape[1]
        
        data_dict['zcrAvg'] = zcrAvg
        
        # Change sign sum
        changeSignArray = librosa.zero_crossings(y+0.01)
        changeSign = np.sum(changeSignArray)
        
        data_dict['changeSign'] = changeSign
        
        # Energia
        hop_length = 1
        frame_length = 1
        
        energy = np.array([
            sum(abs(y[i:i+frame_length]**2))
            for i in range(0, len(y), hop_length)
        ])
        energyValue = sum(energy)
        
        data_dict['energyValue'] = energyValue
        
        # Root-mean-square energy (RMSE) - wartosc 
        rmseValue = math.sqrt(np.mean(y*y))
        
        # Centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        centAvg = cent.sum()/ cent.shape[1]
        
        data_dict['centAvg'] = centAvg
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccsAvg = {'mfcc'+str(idx+1): np.average(mfcc) for idx, mfcc in enumerate(mfccs)}

        data_dict = {**data_dict, **mfccsAvg}
        
        # Klasy
        data_dict['class'] = df_info[df_info['slice_file_name'] == filename]['classID'].values[0]      
        
        dict_list.append(data_dict)
        
        print("Fold %d, %s" % (fold, filename))

# Zapis do csv
df = pd.DataFrame(dict_list)
df.to_csv('sound_features_upd.csv', sep=';')
























