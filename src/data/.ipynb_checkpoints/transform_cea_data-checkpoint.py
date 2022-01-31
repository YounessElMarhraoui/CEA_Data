import os
import sys

import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings("ignore")


class TransformCeaData:
    """
    A class to transform data to have a better representation of data

    ...

    Attributes
    ----------
    data_file_path : str
       the path leading to the data file
    df : dataframe
       the dataframe containing all of the data
    iter_df : dataframe
       the sub-dataframe the contains all of the unique tuples of patients' IDs and their activities

    Methods
    -------
    get_unique_tuples
       Creates a sub-dataframe that contains all of the unique tuples of patients' IDs and their activities
    transform_df
       Transforms the target variable into a binary sequence
    lr_transform_df
        Transforms the dataframe in order to have left data and right data concatenated as rows instead of columns
    """

    def __init__(self, data_file_path):
        """
        Initialize path to the data file

        ...

        Parameters
        ----------
        data_file_path : str
            The path leading to the data file
        """
        self.data_file_path = data_file_path

    def get_unique_tuples(self, df):
        """Creates a sub-dataframe that contains all of the unique tuples of patients' IDs and their activities

        The data is grouped by those two columns in order to get the unique combinations of them

        Parameters
        ----------
        df : dataframe
            The dataframe containing all of the data

        Returns
        -------
        iter_df : dataframe
            A sub-dataframe that contains all of the unique tuples of patients' IDs and their activities
        """
        iter_df = (df[['id_pat', 'activity']]
                   .groupby(['id_pat', 'activity'])
                   .count()
                   .reset_index()
                   )
        return iter_df

    def transform_df(self, df, iter_df):
        """Transforms the target variable into a binary sequence

        Goes through all of the tuples of (id_patient, activity), selects a sub dataframe corresponding to those
        information, transforms the signal peaks into a binary sequence then concatenates all of those dataframes

        Parameters
        ----------
        df : dataframe
            The dataframe containing all of the data
        iter_df : dataframe
            The sub-dataframe that contains all of the unique tuples of patients' IDs and their activities

        Returns
        -------
        transformed_df : dataframe
            A dataframe that contains all of the transformed data
        """
        transformed_df = pd.DataFrame()

        for idx, row in iter_df.iterrows():
            temp_df = df.loc[(df.id_pat == row['id_pat']) & (df.activity == row['activity'])]

            temp_df['appui_leve_droit'] = temp_df.CI_Droit_qui_se_leve - temp_df.CI_Droit_qui_se_pose
            temp_df['appui_leve_gauche'] = temp_df.CI_Gauche_qui_se_leve - temp_df.CI_Gauche_qui_se_pose

            temp_df.appui_leve_droit = temp_df.appui_leve_droit.replace(0, np.nan).ffill().fillna(0)
            temp_df.appui_leve_gauche = temp_df.appui_leve_gauche.replace(0, np.nan).ffill().fillna(0)

            transformed_df = pd.concat([transformed_df, temp_df], axis=0)

        return transformed_df

    def lr_transform_df(self, df):
        """Transforms the dataframe in order to have left data and right data concatenated as rows instead of columns

        Concatenate left data and right data as rows then create a column that specifies if the data is related to
        the right foot or the left foot

        Parameters
        ----------
        df : dataframe
            The dataframe containing all of the data (right and left as columns)

        Returns
        -------
        lr_transformed_df : dataframe
            A dataframe that contains concatenated left and right data in the row axis
        """
        values_replace = {'6min_1': '6min',
                          'GetUpAndGo_2': 'GetUpAndGo',
                          'DepBrasAvant_2': 'DepBrasAvant'}
        df.replace(values_replace, inplace=True)
        df.dropna(inplace=True)

        df.appui_leve_droit = np.maximum(df.appui_leve_droit, 0)
        df.appui_leve_gauche = np.maximum(df.appui_leve_gauche, 0)

        common_cols = ['id_pat', 'date', 'activity', 'timeline']
        right_cols = [col for col in df.columns if col.endswith('_D')] + ['appui_leve_droit']
        left_cols = [col for col in df.columns if col.endswith('_G')] + ['appui_leve_gauche']

        df_right_foot = df[common_cols + right_cols].assign(foot_type=1)
        df_left_foot = df[common_cols + left_cols].assign(foot_type=0)

        df_right_foot.rename(columns={'appui_leve_droit': 'appui_leve'}, inplace=True)
        df_left_foot.rename(columns={'appui_leve_gauche': 'appui_leve'}, inplace=True)

        df_right_foot.rename(
            columns={col: col.replace('_D', '') for col in df_right_foot.columns if col.endswith('_D')},
            inplace=True
        )
        df_left_foot.rename(
            columns={col: col.replace('_G', '') for col in df_left_foot.columns if col.endswith('_G')},
            inplace=True
        )

        lr_transformed_df = pd.concat([df_left_foot, df_right_foot], axis=0)
        return lr_transformed_df


if __name__ == '__main__':
    DATA_FILE_PATH = sys.argv[1]

    td = TransformCeaData(DATA_FILE_PATH)
    df = pd.read_csv(DATA_FILE_PATH)
    iter_df = td.get_unique_tuples(df)
    transformed_df = td.transform_df(df, iter_df)

    transformed_df.to_csv(
        os.path.join(
            '/'.join(DATA_FILE_PATH.split('/')[:-3]),
            "TRANSFORMED_DATA",
            "transformed_cea_data.csv"
        ), header=True, index=False
    )

    lr_transformed_df = td.lr_transform_df(transformed_df)
    lr_transformed_df.to_csv(
        os.path.join(
            '/'.join(DATA_FILE_PATH.split('/')[:-3]),
            "TRANSFORMED_DATA",
            "lr_transformed_cea_data.csv"
        ), header=True, index=False
    )

    print('Process finished!')
