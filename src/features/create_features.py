import os
import pandas as pd
import numpy as np
from src.tools import Config
from dotenv import find_dotenv, load_dotenv
from sklearn import preprocessing

## config file
load_dotenv(find_dotenv())
cfg = Config(project_dir = os.getenv('PROJECT_DIR'), mode = os.getenv('MODE'))

## variables globales
CHEMIN_DATA_INTERIM = os.path.join(cfg.get('directory')['project_dir'], 'data', 'interim')
CHEMIN_TRAIN_EVENTS = os.path.join(CHEMIN_DATA_INTERIM, 'games.csv')
CHEMIN_DATA_PROCESSED = os.path.join(cfg.get('directory')['project_dir'], 'data', 'processed')

## lecture des données
games = pd.read_csv(CHEMIN_TRAIN_EVENTS)


## uniformisation du repère sur chaque match
idx = games[games['away_team_id'] == games['event_team_id']].index
games.loc[idx, ['event_x', 'event_y']] = 100 - games.loc[idx, ['event_x', 'event_y']]

## Indicatrice away / home team
games['features_away_team'] = 0
games.loc[idx, 'features_away_team'] = 1

## Indicatrice pour le changement de possession
idx = (games['event_team_id'] != games['event_team_id'].shift(1)) & (games['game_id'] == games['game_id'].shift(1)) & (games['event_period_id'] == games['event_period_id'].shift(1))
games['features_change_team'] = 0
games.loc[idx, 'features_change_team'] = 1

## features sur le type de d'event
event_type_dummies = pd.get_dummies(games['event_type_id'], prefix = 'event_type')
games = pd.concat([games, event_type_dummies], axis = 1)

games['event_type_id_recoded'] = pd.Categorical(games.event_type_id).codes


## features sur le type de joueur
games['real_position'] = games['real_position'].fillna('Unknown')
player_position_dummies = pd.get_dummies(games['real_position'], prefix = 'position')
player_position_dummies.columns = player_position_dummies.columns.str.lower().str.replace(' ', '_')
games = pd.concat([games, player_position_dummies], axis = 1)
player_position_dummies.head()
games['position_type'] = pd.Categorical(games.real_position).codes

## features sur les zones
games['zone_name_id'] = pd.Categorical(games.zone_name).codes

## Normaliser les positions
games[['event_x', 'event_y']] = (games[['event_x', 'event_y']]) / 100

## Temps entre chaque position
games['features_seconds'] = games['event_seconds_elapsed'] - games['event_seconds_elapsed'].shift(1)
idx =  (games['game_id'] == games['game_id'].shift(1)) & (games['event_period_id'] == games['event_period_id'].shift(1))
games.loc[~idx, 'features_seconds'] = 0

##  ----- features pour determiner qui possede la balle

games_possesion = games.copy()

idx = (games['game_id'] != games['game_id'].shift(1)) | (games['event_period_id'] != games['event_period_id'].shift(1))
games_possesion['changement_possesion'] = games_possesion['features_change_team'].shift(-1)
games_possesion.loc[games_possesion.changement_possesion.isnull(), 'changement_possesion'] = 0
games_possesion.loc[idx, 'changement_possesion'] = 0

# games_possesion_chgt = games_possesion[games_possesion.changement_possesion == 1][['game_id', 'event_period_id', 'event_order']]
# games_possesion_chgt['last_event_possesion'] = games_possesion_chgt.groupby(['game_id', 'event_period_id'])['event_order'].shift(1)
# games_possesion_chgt['last_chgt_possesion'] = games_possesion_chgt['event_order'] - games_possesion_chgt['last_event_possesion']
# games_possesion_chgt['last_chgt_possesion'][games_possesion_chgt.last_chgt_possesion.isnull()] = -1
# print(games_possesion_chgt.head(10))

games_possesion['id_possesion'] = games_possesion.groupby(['game_id', 'event_period_id'])['changement_possesion'].cumsum()
games_possesion['last_chg_possesion_order'] = games_possesion.groupby(['game_id', 'event_period_id', 'id_possesion'])['event_order'].transform('min')
games_possesion['last_chg_possesion_order'] = games_possesion.groupby(['game_id', 'event_period_id'])['last_chg_possesion_order'].shift(1)
games_possesion['last_chg_possesion'] = games_possesion.event_order - games_possesion.last_chg_possesion_order
games_possesion.loc[games_possesion.last_chg_possesion.isnull(), 'last_chg_possesion'] = 0

games_possesion['elapsed_time_since_possesion'] = games_possesion.groupby(['game_id', 'event_period_id', 'last_chg_possesion_order'])['features_seconds'].cumsum()
games_possesion.loc[games_possesion.elapsed_time_since_possesion.isnull(), 'elapsed_time_since_possesion'] = 0

cols_event_t = list(event_type_dummies.columns.values)

cols_event_t1 = [el+'_t1' for el in cols_event_t]

games_possesion[['event_x_t1', 'event_y_t1']] = games_possesion.groupby(['game_id', 'event_period_id'])[['event_x', 'event_y']].shift(1)
games_possesion[['event_x_t2', 'event_y_t2']] = games_possesion.groupby(['game_id', 'event_period_id'])[['event_x', 'event_y']].shift(2)
games_possesion[['event_x_t3', 'event_y_t3']] = games_possesion.groupby(['game_id', 'event_period_id'])[['event_x', 'event_y']].shift(3)
games_possesion[['event_x_t4', 'event_y_t4']] = games_possesion.groupby(['game_id', 'event_period_id'])[['event_x', 'event_y']].shift(4)
games_possesion[['event_x_t5', 'event_y_t5']] = games_possesion.groupby(['game_id', 'event_period_id'])[['event_x', 'event_y']].shift(5)

games_possesion[cols_event_t1] = games_possesion.groupby(['game_id', 'event_period_id'])[cols_event_t].shift(1)

games_possesion['event_type_id_special'] = 0
games_possesion.loc[games_possesion.event_type_id.isin([74, 45, 12, 11, 10, 8, 7, 5]), 'event_type_id_special'] = 1

games_possesion['event_team_id_special'] = 0
games_possesion.loc[games_possesion.event_team_id.isin([152, 1395, 149, 2130, 430]), 'event_team_id_special'] = -1
games_possesion.loc[games_possesion.event_team_id.isin([427, 145, 148, 139]), 'event_team_id_special'] = 1

team_dummies = pd.get_dummies(games_possesion['event_team_id'], prefix = 'team_id')
games_possesion = pd.concat([games_possesion, team_dummies], axis = 1)
cols_teams = list(team_dummies.columns.values)

cols_to_keep = ['game_id', 'event_period_id', 'train', 'changement_possesion', 'elapsed_time_since_possesion', 'last_chg_possesion', 'event_x', 'event_y', 'event_type_id_special', 'event_team_id_special', 'event_type_id', 'event_team_id']
cols_to_keep = cols_to_keep + cols_event_t + cols_event_t1 + ['event_x_t1', 'event_x_t2', 'event_x_t3', 'event_x_t4', 'event_x_t5', 'event_y_t1', 'event_y_t2', 'event_y_t3', 'event_y_t4', 'event_y_t5'] + cols_teams

games_possesion = games_possesion.loc[games_possesion.event_order > 1][cols_to_keep]
games_possesion = games_possesion.loc[~games_possesion.event_x_t1.isnull()]
games_possesion = games_possesion.loc[~games_possesion.event_x_t2.isnull()]
games_possesion = games_possesion.loc[~games_possesion.event_x_t3.isnull()]
games_possesion = games_possesion.loc[~games_possesion.event_x_t4.isnull()]
games_possesion = games_possesion.loc[~games_possesion.event_x_t5.isnull()]

min_max_scaler = preprocessing.MinMaxScaler()
games_possesion['elapsed_time_since_possesion'] = min_max_scaler.fit_transform(games_possesion[['elapsed_time_since_possesion']].values)
games_possesion['last_chg_possesion'] = min_max_scaler.fit_transform(games_possesion[['last_chg_possesion']].values)


## creation des jeux de possession, avec seulement des 12 et 1 pour le test.
games_possesion_reduit = games_possesion[games_possesion.event_type_id.isin([1, 12])]
idx_a_supprimer = games_possesion_reduit.groupby(['game_id', 'event_period_id']).tail(1).index
games_possesion_reduit = games_possesion_reduit.loc[~games_possesion_reduit.index.isin(idx_a_supprimer)]
games_possesion_reduit = pd.concat([games_possesion_reduit[games_possesion_reduit.train == 0], games_possesion_reduit[games_possesion_reduit.train == 1].sample(frac=.3)], axis = 0)


## sauvegarde

### train
games[games.train == 1].to_csv(os.path.join(CHEMIN_DATA_PROCESSED, 'games_train.csv'), index = False)
games_possesion[games_possesion.train == 1].to_csv(os.path.join(CHEMIN_DATA_PROCESSED, 'games_possesion_train.csv'), index = False)

### test
games[games.train == 0].to_csv(os.path.join(CHEMIN_DATA_PROCESSED, 'games_test.csv'), index = False)
games_possesion[games_possesion.train == 0].to_csv(os.path.join(CHEMIN_DATA_PROCESSED, 'games_possesion_test.csv'), index = False)

### les deux ensemble (on veut apprendre sur les matchs "en cours")
games_possesion_reduit.to_csv(os.path.join(CHEMIN_DATA_PROCESSED, 'games_possesion_reduit_train.csv'), index = False)