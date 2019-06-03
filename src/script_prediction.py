import os
import pandas as pd
import numpy as np
from src.tools import Config, Tools
from dotenv import find_dotenv, load_dotenv
import pickle
import scipy
from src.models import DataLoader, DataLoaderTeamChange
from src.models import PositionModel_embeded, PositionModel_events
from src.models import ModelTrainer

## config file
load_dotenv(find_dotenv())
cfg = Config(project_dir = os.getenv('PROJECT_DIR'), mode = os.getenv('MODE'))

## Variables
CHEMIN_DATA_PROCESSED = os.path.join(cfg.get('directory')['project_dir'], 'data', 'processed')
CHEMIN_MODELS = os.path.join(cfg.get('directory')['project_dir'], 'models')

## Chargement des données
games_test = pd.read_csv(os.path.join(CHEMIN_DATA_PROCESSED, 'games_test.csv'))
games_train = pd.read_csv(os.path.join(CHEMIN_DATA_PROCESSED, 'games_train.csv'))
games_possesion_test = pd.read_csv(os.path.join(CHEMIN_DATA_PROCESSED, 'games_possesion_test.csv'))
games_possesion_train = pd.read_csv(os.path.join(CHEMIN_DATA_PROCESSED, 'games_possesion_train.csv'))
games_possesion_reduit = pd.read_csv(os.path.join(CHEMIN_DATA_PROCESSED, 'games_possesion_reduit_train.csv'))

## Chargement des modèles

### modèle pour la prédiction n+1:
loader = DataLoader(cfg, "simple")
model = PositionModel_embeded(cfg.get("models")['position']['embeded'], loader.len_seq_train, loader.len_seq_pred, loader.output_dim)
model.load(which = 'final_embeded_1_200_epochs')

### modèle pour la prédiction n+2 à n+5:
model_position = PositionModel_events(cfg.get("models")['position']['events'], loader.len_seq_train, loader.len_seq_pred, loader.output_dim)
model_position.load()

### modèle pour prédire le changement d'équipe
with open(os.path.join(CHEMIN_MODELS, 'random_forest_2_pkl'), 'rb') as f:
    rf = pickle.load(f)

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

## tests des deux modèles de prédiction de positions

### n+1:
error = []
for i in range(500):
    X_train, X_test = loader.generate_seqs(games_test, loader.len_seq_train, loader.len_seq_pred)
    position = model.predict(Tools.format_seq(X_train))[0]
    error.append(np.sqrt(np.sum((position - X_test[0])**2)))

print(np.mean(error))

### n+2 à n+5:
error = []
for i in range(500):
    X_train, X_test = loader.generate_seqs(games_test, loader.len_seq_train, 2)
    position = model.predict(Tools.format_seq(X_train))[0]
    
    coords = X_train[0]       
    seconds_mean = np.mean(coords[:-10,3])
    
    fresh_coord = np.array([[position[0], position[1], 0, seconds_mean]])
    coords = np.concatenate((coords[1:,], fresh_coord), axis = 0)
                
    coords_position = np.expand_dims(coords, axis = 0)
    position = model_position.predict(coords_position)[0]
    
    error.append(np.sqrt(np.sum((position - X_test[1])**2)))

print(np.mean(error))

## Boucle pour prédire:

lines = []
for game_id in games_test.game_id.unique():
    for event_period_id in [1,2]:
        games = games_test[(games_test.game_id == game_id) & (games_test.event_period_id == event_period_id)]
        games_possesion = games_possesion_test.loc[(games_possesion_test.game_id == game_id) & (games_possesion_test.event_period_id == event_period_id)].tail(1)

        away_team_id = games['away_team_id'].iloc[0]
        home_team_id = games['home_team_id'].iloc[0]
        event_team_id = games['event_team_id'].iloc[-1]
        event_type_id = games['event_type_id'].iloc[-1]
        old_event_x = games['event_x'].iloc[-1]
        old_event_y = games['event_y'].iloc[-1]
        event_order = games['event_order'].iloc[-1]+1
        seq = Tools.generate_seq(games, 50)
        seq_team = Tools.generate_seq(games, 10)
        
        coords = seq[0]       
        seconds_mean = np.mean(coords[:-3,3])
        
        position = np.clip(model.predict(Tools.format_seq(seq))[0], -0.02, 1.02)
        change_team_pred = rf.predict_proba(games_possesion[cols_x])[0][1]
        
        ## rectification des position si on connait le dernier evenement (sachant que le modele est plutôt correct..)
        ## on corrige la sortie, car on peut être plus précis que le modèle
        if event_type_id == 5:           
            if old_event_x < 0:
                position[0] = 0.049
            elif old_event_x > 1:
                position[0] = 0.948
            if old_event_y > 1:
                position[1] = 1
            elif old_event_y < 0:
                position[1] = 0
                
        ## on corrige le tir manqué
        if event_type_id == 13:
            position[0] = np.round(position[0])
            
        ## on corrige le but
        if event_type_id == 16:
            position[0] = 0.5
            position[1] = 0.5
            
        
        ## cas du changement d'équipe : probas + regles metier
        if event_type_id in [4,5,6]:
            is_team_change = False
        elif event_type_id in [2,13,16,45]:
            is_team_change = True
        else:
            is_team_change = change_team_pred >= .4
            
        ## définir qui joue à la dernière ligne
        is_away_team = event_team_id == away_team_id
        
        ## calcul de l'autre équipe si ca doit changer
        if is_away_team:
            actual_team = away_team_id
            other_team = home_team_id
        else:
            actual_team = home_team_id
            other_team = away_team_id
        
        ## calcul final pour savoir quelle equipe et si on inverse
        if is_team_change:
            event_team_id = other_team
            if is_away_team:
                is_position_inverted = False
            else:
                is_position_inverted = True
        else:
            event_team_id = actual_team
            if is_away_team:
                is_position_inverted = True
            else:
                is_position_inverted = False

        
        position_x = position[0]*100
        position_y = position[1]*100
        
        if is_position_inverted is True:
            position_x = 100 - position_x
            position_y = 100 - position_y
                
        event_type_id_pred = 1
        if event_type_id == 3:
            event_type_id_pred = 45
        elif event_type_id == 15:
            event_type_id_pred = 10
        elif event_type_id == 41:
            event_type_id_pred = 5
        elif event_type_id == 50:
            event_type_id_pred = 7
        elif event_type_id == 13:
            event_type_id_pred = 5
        
    
        line = {
            'game_id': game_id,
            'event_period_id': event_period_id,
            'event_type_id': event_type_id_pred,
            'event_team_id': event_team_id,
            'event_x': position_x,
            'event_y': position_y,
            'event_order': event_order
        }
        lines.append(line)
        
        position_x = position[0]
        position_y = position[1]
        
        for e_o in range(event_order+1, event_order+5):
            
            fresh_coord = np.array([[position_x, position_y, 0, seconds_mean]])
            coords = np.concatenate((coords[1:,], fresh_coord), axis = 0)
                
            coords_position = np.expand_dims(coords, axis = 0)
            pred_position = np.clip(model_position.predict(coords_position)[0], -0.02, 1.02)
            
            position_x = pred_position[0]*100
            position_y = pred_position[1]*100
            
            if is_position_inverted is True:
                position_x = 100 - position_x
                position_y = 100 - position_y
            
            
            line = {
                'game_id': game_id,
                'event_period_id': event_period_id,
                'event_type_id': 0,
                'event_team_id': event_team_id,
                'event_x': position_x,
                'event_y': position_y,
                'event_order': e_o
            }
            position_x = pred_position[0]
            position_y = pred_position[1]

            lines.append(line)
        
        
## sauvegarde des résultats
pd.DataFrame(lines).to_csv(os.path.join(CHEMIN_DATA_PROCESSED, 'resultats_test.csv'), index = False)
