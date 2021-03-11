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
           transforms the target variable into a binary sequence
       """

    def __init__(self, data_file_path):
        """
        Initiate path to the data file

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
    print('Process finished!')
