import os
import pandas as pd
import numpy as np
from src.tools import Config
from dotenv import find_dotenv, load_dotenv


## config file
load_dotenv(find_dotenv())
cfg = Config(project_dir = os.getenv('PROJECT_DIR'), mode = os.getenv('MODE'))

## variables globales
CHEMIN_DATA = os.path.join(cfg.get('directory')['project_dir'], 'data', 'raw')
CHEMIN_TRAIN_EVENTS = os.path.join(CHEMIN_DATA, 'games_train_events.csv')
CHEMIN_TEST_EVENTS = os.path.join(CHEMIN_DATA, 'games_test_events.csv')
CHEMIN_PLAYERS = os.path.join(CHEMIN_DATA, 'players.csv')
CHEMIN_DATA_INTERIM = os.path.join(cfg.get('directory')['project_dir'], 'data', 'interim')

## chargement des données
games_train = pd.read_csv(CHEMIN_TRAIN_EVENTS)
games_test = pd.read_csv(CHEMIN_TEST_EVENTS)

## Concatenation train & test
games_train_reduced = games_train[games_test.columns].copy()
games_train_reduced.loc[:, 'train'] = 1
games_test.loc[:, 'train'] = 0

games = pd.concat([games_train_reduced, games_test], axis = 0)
games = games.sort_values(by = ['game_id', 'event_period_id', 'event_order'], inplace = False)

## création d'un dictionnaire d'équipe
teams = games_train[['away_team_id', 'away_team_name']].drop_duplicates().reset_index(drop = True)
teams.columns = ['team_id', 'team_name']

## création d'un dictionnaire de player
players = pd.read_csv(CHEMIN_PLAYERS)
players = players[['player_id', 'real_position', 'real_position_side', 'teamName']]
players = players.merge(teams, how = 'right', right_on = 'team_name', left_on = 'teamName')

## Données finales
games = pd.merge(games, players, how = 'left', left_on = ['event_player_id', 'event_team_id'], right_on = ['player_id', 'team_id'])

## sauvegarde des données
players.to_csv(os.path.join(CHEMIN_DATA_INTERIM, 'players_teams.csv'), index = False)
games.to_csv(os.path.join(CHEMIN_DATA_INTERIM, 'games.csv'), index = False)

