# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, config, type_loader = 'simple'):

        self.data = None
        self.train_data = None
        self.test_data = None
        self.type_loader = type_loader
        self.config = config

        self.len_seq_train = self.config.get('data_loader')[self.type_loader]['len_seq_train']
        self.len_seq_pred = self.config.get('data_loader')[self.type_loader]['len_seq_pred']
        self.batch_size = self.config.get('data_loader')[self.type_loader]['batch_size']
        self.output_dim = self.config.get('data_loader')[self.type_loader]['output_dim']

        self.read_data()
        self.split_train_test()

    def read_data(self):
        self.data = pd.read_csv(self.config.get('data_loader')[self.type_loader]['data_path'])


    def split_train_test(self):

        np.random.seed = 263131
        liste_game_id = np.unique(self.data['game_id'])
        np.random.shuffle(liste_game_id)

        game_id_train = liste_game_id[0:120]
        game_id_test = liste_game_id[120:]

        self.train_data = self.data[self.data['game_id'].isin(game_id_train)]
        self.test_data = self.data[self.data['game_id'].isin(game_id_test)]


    def generate_seqs(self, games, len_seq_train = 10, len_seq_pred = 1):
        # variables en entrée de la fonction
        # games = games
        # len_seq_train = 10
        # len_seq_pred = 1

        # récupérér len_seq_train + len_seq_pred au hasard
        len_seq = len_seq_train + len_seq_pred

        # restreindre le "games" en choisissant le game_id et la period_id
        game_id_random = np.random.choice(np.unique(games['game_id']))
        period_id_random = np.random.randint(1, 3)

        seq = games[(games['game_id'] == game_id_random) & (games['event_period_id'] == period_id_random)]

        event_order_max = np.max(seq['event_order'])
        event_order_min = np.min(seq['event_order'])

        start_seq = np.random.randint(event_order_min, event_order_max - len_seq)
        end_seq = start_seq + len_seq

        seq_sel = seq[(seq['event_order'] >= start_seq) & (seq['event_order'] < end_seq)]
        seq_sel = seq_sel.sort_values(by = ['game_id', 'event_period_id', 'event_order'], inplace = False)
        
        seq_train_coord = np.array(seq_sel.head(len_seq_train)[['event_x', 'event_y', 'features_change_team', 'features_seconds']])
        seq_train_position = np.array(seq_sel.head(len_seq_train)[['position_type']]).flatten()
        seq_train_event = np.array(seq_sel.head(len_seq_train)[['event_type_id_recoded']]).flatten()
        seq_train_zone = np.array(seq_sel.head(len_seq_train)[['zone_name_id']]).flatten()
        

        seq_pred = np.array(seq_sel.tail(len_seq_pred)[['event_x', 'event_y']])


        return [seq_train_coord, seq_train_position, seq_train_event, seq_train_zone], seq_pred


    def generator(self, type = "train"):
        data = self.train_data if type == "train" else self.test_data
        while True:
            X_COORD = np.zeros((self.batch_size, self.len_seq_train, 4))
            X_POSITION = np.zeros((self.batch_size, self.len_seq_train))
            X_EVENT = np.zeros((self.batch_size, self.len_seq_train))
            X_ZONE = np.zeros((self.batch_size, self.len_seq_train))
            
            Y = np.zeros((self.batch_size, self.output_dim))
            
            for i in range(self.batch_size):
                X_train, Y_train = self.generate_seqs(data, self.len_seq_train, self.len_seq_pred)

                X_COORD[i, : ] = X_train[0]
                X_POSITION[i, : ] = X_train[1]
                X_EVENT[i, : ] = X_train[2]
                X_ZONE[i, : ] = X_train[3]
                Y[i, : ] = Y_train
        
            yield [X_COORD, X_POSITION, X_EVENT, X_ZONE], Y


class DataLoaderEvents(DataLoader):

    def generator(self, type = "train"):
        data = self.train_data if type == "train" else self.test_data
        while True:
            X_COORD = np.zeros((self.batch_size, self.len_seq_train, 4))
            
            Y = np.zeros((self.batch_size, self.output_dim))
            
            for i in range(self.batch_size):
                X_train, Y_train = self.generate_seqs(data, self.len_seq_train, self.len_seq_pred)

                X_COORD[i, : ] = X_train[0]
                Y[i, : ] = Y_train
        
            yield X_COORD, Y

class DataLoaderTeamChange(DataLoader):

    def __init__(self, config, type_loader = 'simple'):

        self.data = None
        self.train_data = None
        self.test_data = None
        self.type_loader = type_loader
        self.config = config

        self.input_dim = self.config.get('data_loader')[self.type_loader]['input_dim']
        self.batch_size = self.config.get('data_loader')[self.type_loader]['batch_size']
        self.output_dim = self.config.get('data_loader')[self.type_loader]['output_dim']

        self.read_data()
        self.split_train_test()

    # def read_data(self):
    #     data = pd.read_csv(self.config.get('data_loader')[self.type_loader]['data_path'])
    #     idx_a_predire = data.groupby(['game_id', 'event_period_id']).tail(1).index
    #     self.data = data.loc[~data.index.isin(idx_a_predire)]


    def split_train_test(self):

        np.random.seed = 3216321

        self.train_data = self.data.sample(frac = .75)
        self.test_data = self.data.loc[~self.data.index.isin(self.train_data.index)]

    def generate_seqs(self, games):

        # restreindre le "games" en choisissant le game_id et la period_id
        np.random.seed = None
        line_random = np.random.choice(np.unique(games.index))

        line = games.loc[line_random]

        cols_x = ['elapsed_time_since_possesion', 'last_chg_possesion', 'event_x',
       'event_y', 'event_type_1', 'event_type_2', 'event_type_3',
       'event_type_4', 'event_type_5', 'event_type_6', 'event_type_7',
       'event_type_8', 'event_type_10', 'event_type_11', 'event_type_12',
       'event_type_13', 'event_type_14', 'event_type_15', 'event_type_16',
       'event_type_41', 'event_type_44', 'event_type_45', 'event_type_49',
       'event_type_50', 'event_type_51', 'event_type_54', 'event_type_55',
       'event_type_61', 'event_type_74', 'event_type_1_t1',
       'event_type_2_t1', 'event_type_3_t1', 'event_type_4_t1',
       'event_type_5_t1', 'event_type_6_t1', 'event_type_7_t1',
       'event_type_8_t1', 'event_type_10_t1', 'event_type_11_t1',
       'event_type_12_t1', 'event_type_13_t1', 'event_type_14_t1',
       'event_type_15_t1', 'event_type_16_t1', 'event_type_41_t1',
       'event_type_44_t1', 'event_type_45_t1', 'event_type_49_t1',
       'event_type_50_t1', 'event_type_51_t1', 'event_type_54_t1',
       'event_type_55_t1', 'event_type_61_t1', 'event_type_74_t1',
       'event_x_t1', 'event_y_t1',
       'event_type_id_special', 'event_team_id_special']

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
       'team_id_1395', 'team_id_2128', 'team_id_2130']

        # cols_x = ['elapsed_time_since_possesion', 'last_chg_possesion', 'event_x', 'event_y', 'event_x_t1', 'event_y_t1']

        line_x = np.array(line[cols_x])
        # line_x = np.array(line[['event_x', 'event_y']])
        line_y = np.array(line[['changement_possesion']])

        return line_x, line_y

    def generator(self, type = "train"):
        data = self.train_data if type == "train" else self.test_data

        while True:
            X = np.zeros((self.batch_size, self.input_dim))
            Y = np.zeros((self.batch_size, self.output_dim))
            
            for i in range(self.batch_size):
                X_train, Y_train = self.generate_seqs(data)

                X[i, : ] = X_train
                Y[i, : ] = Y_train
        
            yield X, Y


# class DataLoaderTeamChange(DataLoader):

#     def generate_seqs(self, games, len_seq_train = 10, len_seq_pred = 1):
#         # variables en entrée de la fonction
#         # games = games
#         # len_seq_train = 10
#         # len_seq_pred = 1

#         # récupérér len_seq_train + len_seq_pred au hasard
#         len_seq = len_seq_train + len_seq_pred

#         # restreindre le "games" en choisissant le game_id et la period_id
#         game_id_random = np.random.choice(np.unique(games['game_id']))
#         period_id_random = np.random.randint(1, 3)

#         seq = games[(games['game_id'] == game_id_random) & (games['event_period_id'] == period_id_random)]

#         event_order_max = np.max(seq['event_order'])
#         event_order_min = np.min(seq['event_order'])

#         start_seq = np.random.randint(event_order_min, event_order_max - len_seq)
#         end_seq = start_seq + len_seq

#         seq_sel = seq[(seq['event_order'] >= start_seq) & (seq['event_order'] < end_seq)]
#         seq_sel = seq_sel.sort_values(by = ['game_id', 'event_period_id', 'event_order'], inplace = False)
        
#         seq_train_coord = np.array(seq_sel.head(len_seq_train)[['event_x', 'event_y', 'features_change_team', 'features_seconds']])
#         seq_train_position = np.array(seq_sel.head(len_seq_train)[['position_type']]).flatten()
#         seq_train_event = np.array(seq_sel.head(len_seq_train)[['event_type_id_recoded']]).flatten()
#         seq_train_zone = np.array(seq_sel.head(len_seq_train)[['zone_name_id']]).flatten()
        

#         seq_pred = np.array(seq_sel.tail(len_seq_pred)[['features_change_team']])


#         return [seq_train_coord, seq_train_position, seq_train_event, seq_train_zone], seq_pred