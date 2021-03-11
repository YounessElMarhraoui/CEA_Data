import os
import glob
import sys

import sqlite3

import pandas as pd


class StructureCeaData:
    """
    A class to structure data from folders into one file

    ...

    Attributes
    ----------
    data_path : str
        the path leading to the folder that contains data sub-folders
    date : str
        the date in string format to be formatted into datetime
    file_path : str
        the path of csv file to read and to which new columns are added based on its hierarchical position
    df : dataframe
        the final dataframe to be stored

    Methods
    -------
    format_date
        Transforms a String into a formatted datetime
    custom_read_csv
        Reads a csv file and adds new columns that are specific to to this file
    create_global_df
        Lists all existing csv files in sub-folders and concatenates them into one global dataframe
    store_df
        Stores dataframe into csv, parquet and database formatted files
    """

    def __init__(self, data_path):
        """
        Initiate path to raw data as well as for structured data

        ...

        Parameters
        ----------
        data_path : str
            The path leading to the folder that contains data sub-folders
        """
        self.data_path = data_path
        self.raw_data_path = os.path.join(self.data_path, 'RAW_DATA', 'DATA_CEA_Activity')
        self.structured_data_path = os.path.join(self.data_path, 'STRUCTURED_DATA', 'DATA_CEA')

    def format_date(self, date):
        """Transforms a String into a formatted datetime

        If the String is corresponding to the default format, it will be transformed into a datetime variable.
        Otherwise, it will be normalized before formatting

        Parameters
        ----------
        date : str
            The String that has to be formatted into datetime

        Returns
        -------
        formatted_date : datetime
            A formatted datetime variable
        """
        try:
            formatted_date = pd.to_datetime(date, format='%d %b %Y')
        except:
            month_names = {'janvier': 'Jan', 'février': 'Feb',
                           'fév': 'Feb', 'fev': 'Feb',
                           'mars': 'Mar', 'avril': 'Apr',
                           'avr': 'Apr', 'mai': 'May',
                           'juin': 'Jun', 'juillet': 'Jul',
                           'juil': 'Jul', 'aout': 'Aug',
                           'août': 'aout', 'septembre': 'Sep',
                           'sept': 'Sep', 'octobre': 'Oct',
                           'novembre': 'Nov', 'décembre': 'Dec',
                           'déc': 'Dec'
                           }

            day = date.split()[0]
            month = date.split()[1].lower()
            year = date.split()[2]

            if month not in month_names.values() and month not in [ele.lower() for ele in list(month_names.values())]:
                month = month_names[month]

            formatted_date = pd.to_datetime(' '.join([day, month, year]), format='%d %b %Y')
        return formatted_date

    def custom_read_csv(self, file_path):
        """Reads a csv file and adds new columns that are specific to to this file

        Extracts new information to add to each file from the folder names then reads the csv file add assigns new
        columns with those new information

        Parameters
        ----------
        file_path : str
            The path of csv file to read and to which new columns are added based on its hierarchical position

        Returns
        -------
        df : dataframe
            A dataframe that contains data from the csv file as well as new columns of the additional information
        """
        # Values to keep are folder names referring to patient ID, date and the activity
        values_to_keep = file_path.split('\\')[-4:-1]
        df = (pd.read_csv(file_path)
                .assign(
                    id_pat=values_to_keep[0],
                    date=self.format_date(values_to_keep[1]),
                    activity=values_to_keep[2],
                )
            )
        return df

    def create_global_df(self):
        """Lists all existing csv files in sub-folders and concatenates them into one global dataframe

        Returns
        -------
        df : dataframe
            A global dataframe that contains data from all of the existing csv files
        """
        print('Listing files... (1/4)')
        list_files = glob.glob(self.raw_data_path + "/**/*.csv", recursive=True)
        print('Concatenating dataframes... (2/4)')
        df = pd.concat(map(self.custom_read_csv, list_files))

        real_cols = [col for col in df.columns if df[col].dtype == 'float64']
        txt_cols = [col for col in df.columns if df[col].dtype in ['object', 'datetime64[ns]']]

        df = df.reindex(columns=txt_cols + real_cols)
        return df

    def store_df(self, df):
        """Stores dataframe into csv, parquet and database formatted files

        Creates a connection with a database file (creates it if doesn't exist), then drops the table if
        it already exists. At last, it stores the data into this database file as well as in CSV and parquet formats

        Parameters
        ----------
        df : dataframe
            the final dataframe to be stored
        """
        print('Connecting to database... (3/4)')
        conn = sqlite3.connect(os.path.join(self.structured_data_path, 'cea_data.db'))
        cur = conn.cursor()

        cur.execute('DROP TABLE IF EXISTS cea_data')

        print('Storing data... (4/4)')
        df.to_sql('cea_data', conn, if_exists='replace')
        conn.close()
        print('     SQL storage (DONE - 1/3)')
        df.to_csv(os.path.join(self.structured_data_path, 'all_cea_data.csv'), index=False)
        print('     CSV storage (DONE - 2/3)')
        df.to_parquet(os.path.join(self.structured_data_path, 'all_cea_data.parquet.gzip'), compression='gzip')
        print('     Parquet storage (DONE - 3/3)')


if __name__ == '__main__':
    DATA_PATH = sys.argv[1]

    sd = StructureCeaData(DATA_PATH)
    global_df = sd.create_global_df()
    sd.store_df(global_df)
    print('Finished!')
