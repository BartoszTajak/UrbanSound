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




# Load data from file
path = r'csv\UrbanSound8K.csv'
df_info = pd.read_csv(path)


def base_features():
    ''' Base function to extract features from audio files

        save all values to CSV files
    '''
    dict_list = []
    numer =1
    # iteration over all dictionary
    for fold in range(1, 11):
        # iteration over all files in dictionary (fold)
        for filename in os.listdir(r'archive\fold'+str(fold)):
            # load sound by librosa
            y, sr = librosa.load((r'archive\fold'+str(fold)+'\\'+filename), sr = 44100, mono = True)

            # empty dictionary
            data_dict = {}

            # ZCR
            zcr = librosa.feature.zero_crossing_rate(y+0.01)
            zcrAvg = np.sum(zcr) / zcr.shape[1]
            data_dict['zcrAvg'] = zcrAvg

            # Change sign sum
            changeSignArray = librosa.zero_crossings(y+0.01)
            changeSign = np.sum(changeSignArray)
            data_dict['changeSign'] = changeSign

            # Energy
            hop_length = 1
            frame_length = 1
            energy = np.array([sum(abs(y[i:i+frame_length]**2))for i in range(0, len(y), hop_length) ])
            energyValue = sum(energy)
            data_dict['energyValue'] = energyValue

            # Root-mean-square energy (RMSE) - value
            rmseValue = math.sqrt(np.mean(y*y))
            data_dict['rmseValue'] = rmseValue

            # Centroid
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            centAvg = cent.sum()/ cent.shape[1]
            data_dict['centAvg'] = centAvg

            # MFCC
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfccsAvg = {'mfcc'+str(idx+1): np.average(mfcc) for idx, mfcc in enumerate(mfccs)}

            data_dict = {**data_dict, **mfccsAvg}

            # classes
            data_dict['class'] = df_info[df_info['slice_file_name'] == filename]['classID'].values[0]

            dict_list.append(data_dict)

            print("%r, Fold %d, %s" % (numer,fold, filename))
            numer +=1


    # save to  csv
    df = pd.DataFrame(dict_list)
    df.to_csv(r'csv\sound_features_upd.csv', sep=';')

def extended_features():
    ''' additional function to extract features from audio files

        save all values to CSV files
    '''

    dict_list = []
    numer =1
    # iteration over all dictionary
    for fold in range(1, 11):
        # iteration over all files in dictionary (fold)
        for filename in os.listdir(r'archive\fold'+str(fold)):
            # load sound by librosa
            y, sr = librosa.load((r'archive\fold'+str(fold)+'\\'+filename), sr = 44100, mono = True)

            # empty dictionary
            data_dict = {}
            try:
                #delta_mfccs
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                delta_mfccs = librosa.feature.delta(mfccs)
                mfccsAvg_delta = {'mfcc_delta'+str(idx+1): np.average(delta_mfccs) for idx, delta_mfccs in enumerate(delta_mfccs)}

                # chroma_stft
                chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                chroma_stftAvg = {'chroma_stft' + str(idx + 1): np.average(chroma_stft) for idx, chroma_stft in enumerate(chroma_stft)}

                # rolloff
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                rolloffAvg = np.mean(rolloff)
                data_dict['rolloff'] = rolloffAvg

                print("%r, Fold %d, %s" % (numer,fold, filename))
                numer +=1
            except:
                print('fail')
                mfccsAvg_delta = {'mfcc_delta1': 0}


            data_dict = {**data_dict, **mfccsAvg_delta,**chroma_stftAvg}

            # classes
            data_dict['class'] = df_info[df_info['slice_file_name'] == filename]['classID'].values[0]
            dict_list.append(data_dict)

    # save to  csv
    df = pd.DataFrame(dict_list)
    df.to_csv(r'csv\sound_features_upd_delta_mfccs.csv', sep=';')

def extended_features_part_2():
    ''' additional function to extract features from audio files

        save all values to CSV files
    '''

    dict_list = []
    numer =1
    # iteration over all dictionary
    for fold in range(1, 11):
        # iteration over all files in dictionary (fold)
        for filename in os.listdir(r'archive\fold'+str(fold)):
            # load sound by librosa
            y, sr = librosa.load((r'archive\fold'+str(fold)+'\\'+filename), sr = 44100, mono = True)

            # empty dictionary
            data_dict = {}
            try:
                #spectral_flux
                tempo_y,_= librosa.beat.beat_track(y, sr=sr)
                data_dict['spectral_flux'] = tempo_y

                ###spectral_flux
                onset_env = librosa.onset.onset_strength(y, sr=sr)
                onset_env = np.mean(onset_env)
                data_dict['spectral_flux'] = onset_env

                # Spectral Bandwidth
                spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(y, sr=sr)[0]
                spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(y, sr=sr, p=3)[0]
                spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(y, sr=sr, p=4)[0]
                data_dict['spectral_bandwidth_2'] = np.mean(spectral_bandwidth_2)
                data_dict['spectral_bandwidth_3'] = np.mean(spectral_bandwidth_3)
                data_dict['spectral_bandwidth_4'] = np.mean(spectral_bandwidth_4)


                print("%r, Fold %d, %s" % (numer,fold, filename))
                numer +=1
            except:
                print('fail')
                data_dict['0'] = 'error'


            # classes
            data_dict['class'] = df_info[df_info['slice_file_name'] == filename]['classID'].values[0]
            dict_list.append(data_dict)

    # save to  csv
    df = pd.DataFrame(dict_list)
    df.to_csv(r'csv\sound_features_extended_part_2.csv', sep=';')

def clean_data():
    ''' function to clean and convert data

        save all vales to csv files
    '''
    # Base features
    path = r'csv\sound_features_upd.csv'
    df_feature = pd.read_csv(path, sep=';',index_col=0)

    # Extended features
    path = r'csv\sound_features_extended.csv'
    df_feature_ext = pd.read_csv(path, sep=';',index_col=0)

    # Extended features r02
    path = r'csv\sound_features_extended_part_2.csv'
    df_feature_ext_part_2 = pd.read_csv(path, sep=';',index_col=0)

    # Combine two dataframes
    df_all=[df_feature,df_feature_ext,df_feature_ext_part_2]
    df_new = pd.concat(df_all,axis=1)

    # remove empty rows
    df_new.dropna(inplace=True)

    # remove duplicated columns
    df_new = df_new.loc[:,~df_new.columns.duplicated()]
    df_new['class'] = df_new['class'].astype('str')

    # remove rows with value 0 or 1
    for i in df_new.columns:
     df_new.drop(df_new[df_new[i] == 0 | 1 ].index, inplace=True)


    # set new index
    df_new.index = np.arange(1, len(df_new) + 1)
    df_new.to_csv(r'csv\sound_features_all.csv', sep=';')




base_features()
extended_features()
extended_features_part_2()
clean_data()

