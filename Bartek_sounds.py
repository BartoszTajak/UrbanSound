
import xgboost
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.fft import fft
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.layers import Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
# Load data from file



class Sound_Models():



    def base_features(self):
        ''' Base function to extract features from audio files

            save all values to CSV files
        '''

        path = r'csv\UrbanSound8K.csv'
        df_info = pd.read_csv(path)

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

    def extended_features(self):
        ''' additional function to extract features from audio files

            save all values to CSV files
        '''

        path = r'csv\UrbanSound8K.csv'
        df_info = pd.read_csv(path)

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

    def extended_features_part_2(self):
        ''' additional function to extract features from audio files

            save all values to CSV files
        '''

        path = r'csv\UrbanSound8K.csv'
        df_info = pd.read_csv(path)

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

    def clean_data(self):
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

    def Correlation_and_split_data(self):

        df = pd.read_csv(r'csv\sound_features_all.csv', sep=';' , index_col=0)
        df_corr = df.drop(columns=['class'])
        corr = df_corr.corr()
        # sns.heatmap(corr, annot=True)
        # plt.show()
        corr.to_csv(r'csv\sound_corr.csv', sep=';')
        df = pd.read_csv(r'csv\sound_features_all.csv', sep=';', index_col=0)
        X = df.drop(columns=['class', 'changeSign', 'spectral_bandwidth_4','rolloff','mfcc22'])
        Y = df['class']

        X_train, X_test, self.y_train, self.ytest = train_test_split(X, Y, test_size=0.2)
        min_max_scaler = MinMaxScaler()
        self.x_train_norm = min_max_scaler.fit_transform(X_train)
        self.x_test_norm = min_max_scaler.transform(X_test)

    def Tree_Classification(self):

        tree_model = tree.DecisionTreeClassifier()
        tree_model.fit(self.x_train_norm, self.y_train)
        y_tree_pred = tree_model.predict(self.x_test_norm)
        con_tree = confusion_matrix(self.ytest, y_tree_pred)
        print(classification_report(self.ytest, y_tree_pred))
        print(con_tree)

        print(matthews_corrcoef(self.ytest, y_tree_pred))
        print(cohen_kappa_score(self.ytest, y_tree_pred))

        y_pred_proba = tree_model.predict_proba(self.x_test_norm)
        score = roc_auc_score(self.ytest, y_pred_proba, multi_class='ovr', average='weighted')
        print(score)

    def Xgboost_Classification(self):

        xgboost_c = xgboost.XGBClassifier(max_depth=4,n_estimators=200,learning_rate=0.2,min_child_weight=2,subsample=0.9,max_delta_step=1)
        xgboost_c.fit(self.x_train_norm, self.y_train)
        y_xgb = xgboost_c.predict(self.x_test_norm)
        con_xgb = confusion_matrix(self.ytest,y_xgb )
        print(con_xgb)
        print(classification_report(self.ytest,y_xgb))

        print(matthews_corrcoef(self.ytest, y_xgb))
        print(cohen_kappa_score(self.ytest, y_xgb))

        y_pred_proba = xgboost_c.predict_proba(self.x_test_norm)
        print(roc_auc_score(self.ytest, y_pred_proba, multi_class='ovo'))

    def RandomForest_Clasification(self):

        rnd_clf = RandomForestClassifier(max_depth=8,n_estimators=500, max_leaf_nodes=16,bootstrap=True, n_jobs=1)
        rnd_clf.fit(self.x_train_norm, self.y_train)
        y_pred_rf = rnd_clf.predict(self.x_test_norm)
        con_rf = confusion_matrix(self.ytest,y_pred_rf )
        print(classification_report(self.ytest,y_pred_rf))
        print(con_rf)

    def SVC_Clasification(self):


        svm_clf = SVC(kernel = 'rbf', C=900, gamma=2, coef0=1.0,probability=True)
        svm_clf.fit(self.x_train_norm, self.y_train)
        SVC_CL = svm_clf.predict(self.x_test_norm)
        con_rf = confusion_matrix(self.ytest, SVC_CL)
        print(con_rf)
        print(classification_report(self.ytest, SVC_CL))

        print(matthews_corrcoef(self.ytest, SVC_CL))
        print(cohen_kappa_score(self.ytest, SVC_CL))

        y_pred_proba = svm_clf.predict_proba(self.x_test_norm)
        score = roc_auc_score(self.ytest, y_pred_proba, multi_class='ovo')
        print(score)

    def Neutral_Network(self):

        mypath = r'models\NeuralNetwork'
        arr = os.listdir(mypath)
        if 'UrbanSound' in arr:
            model = tf.keras.models.load_model(r'C:\Users\barto\PycharmProjects\UrbanSound\models\NeuralNetwork\UrbanSound')
            Y_predictions_test = model.predict(self.x_test_norm)
            con_tree = confusion_matrix(self.ytest, np.argmax(Y_predictions_test, axis=1))
            print(con_tree)
            print(classification_report(self.ytest, np.argmax(Y_predictions_test, axis=1)))

            print(matthews_corrcoef(self.ytest, np.argmax(Y_predictions_test, axis=1)))
            print(cohen_kappa_score(self.ytest, np.argmax(Y_predictions_test, axis=1)))

        else:
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Flatten(input_shape=(self.x_train_norm.shape[1], 1)))
            model.add(tf.keras.layers.Dense(1000, activation='tanh'))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(1000, activation='relu', kernel_initializer='he_normal'))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(1000, activation='relu', kernel_initializer='he_normal'))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(1000, activation='relu', kernel_initializer='he_normal'))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(1000, activation='relu'))
            model.add(tf.keras.layers.Dense(10, activation='softmax'))
            model.compile(optimizer='sgd', loss='SparseCategoricalCrossentropy',metrics='accuracy')
            model.fit(self.x_train_norm, self.y_train, epochs=300, verbose='auto',batch_size=16)
            Y_predictions_test = model.predict(self.x_test_norm)

            con_tree = confusion_matrix(self.ytest, np.argmax(Y_predictions_test, axis=1))
            print(con_tree)
            print(classification_report(self.ytest, np.argmax(Y_predictions_test, axis=1)))
            print(matthews_corrcoef(self.ytest, Y_predictions_test))
            print(cohen_kappa_score(self.ytest, Y_predictions_test))
            model.save(r'models\NeuralNetwork\UrbanSound')

    def SVC_clas_test(self):
        df = pd.read_csv(r'csv\sound_features_all.csv', sep=';', index_col=0)
        # X = df.drop(columns=['class', 'changeSign', 'spectral_bandwidth_4', 'spectral_bandwidth_3', 'rolloff', 'mfcc22'])
        X = df.drop(columns=['class'])
        Y = df['class']
        X_train, X_test, y_train, ytest = train_test_split(X, Y, test_size=0.2)
        min_max_scaler = MinMaxScaler()
        x_train_norm = min_max_scaler.fit_transform(X_train)
        x_test_norm = min_max_scaler.transform(X_test)

        from sklearn.model_selection import GridSearchCV

        xgboost_c = xgboost.XGBClassifier()
        param_grid={ 'max_depth' : [3,4,5,6],'learning_rate' : [0.1,0.2,0.3],'n_estimators': [150,200,250,300],'min_child_weight':[1,2,3]
            ,'subsample':[0.8,0.9,1],'max_delta_step':[1,2,3]}

        grid = GridSearchCV(xgboost_c,param_grid)
        grid.fit(x_train_norm, y_train)
        print(grid.best_params_)

        # svm_clf.fit(x_train_norm, y_train)
        # SVC_CL = svm_clf.predict(x_test_norm)
        # con_rf = confusion_matrix(ytest, SVC_CL)
        # print(classification_report(ytest, SVC_CL))
        # print(con_rf)



Exep1 = Sound_Models()
Exep1.Correlation_and_split_data()
# Exep1.Tree_Classification()
# Exep1.Xgboost_Classification()
# Exep1.RandomForest_Clasification()
Exep1.SVC_Clasification()
# Exep1.Neutral_Network()


