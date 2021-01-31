from helpers import file_to_df, df_to_file, load_model, standarize_data, pd
from traceback import print_exc

if __name__ == '__main__':
    try:
        loaded_model = load_model('tree_final_model.sav')
        df_original = file_to_df()
        df = df_original.copy(deep=True)
        df.drop(columns="Unnamed: 0", inplace=True)
        df.drop_duplicates(subset=None, inplace=True)
        ## preprocesing on data
        df.drop(columns=['demo_id', 'demo_round_id', 'weapon_fire_id'], inplace=True)


        df['team'] = pd.get_dummies(df['team'])
        df['map_name'] = pd.get_dummies(df['map_name'])
        df = pd.concat([df, pd.get_dummies(df['TYPE'])], axis=1).drop(columns=['TYPE'])

        df['throw_detonate_time'] = df['detonation_tick'] / 128 - df['throw_tick'] / 128

        try:
            df = df.drop(index=[df[(df['flashbang'] == 1) & (df['throw_detonate_time'] > 3)].index[0]])
        except Exception:
            pass

        start_x_point = df['throw_from_raw_x']
        start_y_point = df['throw_from_raw_y']
        start_z_point = df['throw_from_raw_z']
        end_x_point = df['detonation_raw_x']
        end_y_point = df['detonation_raw_y']
        end_z_point = df['detonation_raw_z']

        traveled_length_3D = ((start_x_point - end_x_point) ** 2 + (end_y_point - start_y_point) ** 2 + (
                end_z_point - start_z_point) ** 2) ** 0.5

        traveled_length_2D = ((start_x_point - end_x_point) ** 2 + (end_y_point - start_y_point) ** 2) ** 0.5

        df['pseudo_angle'] = traveled_length_2D / traveled_length_3D

        pseudo_velocity = traveled_length_3D / df['throw_detonate_time']
        df['pseudo_velocity'] = pseudo_velocity
        df['traveled_length_3D'] = traveled_length_3D
        df = df[df['pseudo_velocity'] < 1200]
        df.drop(
            columns=['detonation_raw_z', 'detonation_raw_y', 'detonation_raw_x', 'throw_from_raw_z', 'throw_from_raw_y',
                     'throw_from_raw_x'], inplace=True)
        df.reset_index(inplace=True, drop=True)

        df.drop(columns=['LABEL'], inplace=True)
        df.drop(columns=['throw_tick', 'detonation_tick', 'throw_detonate_time'], inplace=True)
        df_standarized = standarize_data(df)
        result = loaded_model.predict(df_standarized)
        df['RESULT'] = result.astype(bool)
        df_to_file(df_original,df)
    except Exception as e:
        print_exc()
        print(e)
