import os
import numpy as np

class Tools:
    @staticmethod
    def create_dir_if_not_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @staticmethod
    def generate_seq(seq, len_seq):

        seq = seq.sort_values(by = ['game_id', 'event_period_id', 'event_order'], inplace = False)

        seq_sel = seq.tail(len_seq)

        seq_train_coord = np.array(seq_sel[['event_x', 'event_y', 'features_change_team', 'features_seconds']])
        seq_train_position = np.array(seq_sel[['position_type']]).flatten()
        seq_train_event = np.array(seq_sel[['event_type_id_recoded']]).flatten()
        seq_train_zone = np.array(seq_sel[['zone_name_id']]).flatten()

        return [seq_train_coord, seq_train_position, seq_train_event, seq_train_zone]

    @staticmethod
    def format_seq(x, batch_size = 1):
        coord, position, event, zone = x
        len_seq_train = coord.shape[0]
        coord_shape_y = coord.shape[1]

        X_COORD = np.zeros((batch_size, len_seq_train, 4))
        X_POSITION = np.zeros((batch_size, len_seq_train))
        X_EVENT = np.zeros((batch_size, len_seq_train))
        X_ZONE = np.zeros((batch_size, len_seq_train))

        for i in range(batch_size):
            X_COORD[i, : ] = coord
            X_POSITION[i, : ] = position
            X_EVENT[i, : ] = event
            X_ZONE[i, : ] = zone
        
        return [X_COORD, X_POSITION, X_EVENT, X_ZONE]
    
    @staticmethod
    def format_seq_team(x):
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

        line_x = np.array(x[cols_x])

        return line_x
