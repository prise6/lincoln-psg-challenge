
import os
import pandas as pd
import numpy as np
from src.tools import Config
from dotenv import find_dotenv, load_dotenv
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle


## config file
load_dotenv(find_dotenv())
cfg = Config(project_dir = os.getenv('PROJECT_DIR'), mode = os.getenv('MODE'))

## variales globales du script
CHEMIN_DATA_PROCESSED = os.path.join(cfg.get('directory')['project_dir'], 'data', 'processed')

## import des données
games_possesion_train = pd.read_csv(os.path.join(CHEMIN_DATA_PROCESSED, 'games_possesion_train.csv'))
print(games_possesion_train.shape)
games_possesion_reduit = pd.read_csv(os.path.join(CHEMIN_DATA_PROCESSED, 'games_possesion_reduit_train.csv'))
print(games_possesion_reduit.shape)

## Decoupage train/test/valid
np.random.seed = 6584254
rf_train = games_possesion_train.sample(frac=.7)
rf_train = rf_train[rf_train.event_type_id.isin([1,12])]

rf_test = games_possesion_train.iloc[~games_possesion_reduit.index.isin(rf_train.index)]
rf_test = rf_test[rf_test.event_type_id.isin([1,12])]

rf_valid = games_possesion_reduit[games_possesion_reduit.event_type_id.isin([1, 12])]

np.random.seed = None

## colonnes 
cols_x = ['elapsed_time_since_possesion', 'last_chg_possesion', 'event_x',
       'event_y', 'event_type_id_special', 'event_team_id_special',
       'event_type_1',
       'event_type_12',
       'event_type_1_t1', 'event_type_2_t1', 'event_type_3_t1',
       'event_type_4_t1', 'event_type_5_t1', 'event_type_6_t1',
       'event_type_7_t1', 'event_type_8_t1', 'event_type_10_t1',
       'event_type_11_t1', 'event_type_12_t1', 'event_type_13_t1',
       'event_type_14_t1', 'event_type_15_t1', 'event_type_16_t1',
       'event_type_41_t1', 'event_type_44_t1', 'event_type_45_t1',
       'event_type_49_t1', 'event_type_50_t1', 'event_type_51_t1',
       'event_type_54_t1', 'event_type_55_t1', 'event_type_61_t1',
       'event_type_74_t1', 'event_x_t1', 'event_y_t1', 'team_id_139',
       'team_id_140', 'team_id_143', 'team_id_144', 'team_id_145',
       'team_id_146', 'team_id_147', 'team_id_148', 'team_id_149',
       'team_id_150', 'team_id_152', 'team_id_427', 'team_id_428',
       'team_id_429', 'team_id_430', 'team_id_694', 'team_id_1028',
       'team_id_1395', 'team_id_2128', 'team_id_2130',
       'event_x_t2', 'event_y_t2',
        'event_x_t3', 'event_y_t3',
         'event_x_t4', 'event_y_t4',
          'event_x_t5', 'event_y_t5',
         ]

## Forêt Aléatoires
rf = RandomForestClassifier(oob_score=True, n_estimators=200)
rf.fit(rf_train[cols_x], y = rf_train['changement_possesion'])

## Validation des résultats
to_test = rf_test

prediction = rf.predict(to_test[cols_x])
proba = rf.predict_proba(to_test[cols_x])
reality = to_test['changement_possesion']
prediction_cutoff = [1 if pred>0.4 else 0 for pred in proba[:, 1]]

## matrice de confusion
A, B = confusion_matrix(prediction_cutoff, reality)
print((A[0]+B[1])/np.sum(A+B))

## plots
# fpr, tpr, thresholds = roc_curve(reality, proba[:,1], pos_label = 1)
# plt.plot(fpr, tpr)

## Importance des variables
importance = pd.DataFrame(rf.feature_importances_)
importance['vars'] = cols_x
print(importance.rename({'0': 'test'}))

## sauvegarde du modèle
# with open('models/random_forest_1_pkl','wb') as f:
#     pickle.dump(rf, f)