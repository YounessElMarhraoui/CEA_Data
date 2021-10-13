import numpy as np
import pandas as pd
from scipy.fft import fft, dct
import scipy.stats as stats


class FixedFrameConstructor:
    """
    A class to generate frames of data based on fixed window size.

    ...

    Attributes
    ----------
    df : dataframe
       the dataframe that contains all of the data
    frame_size : int
       the size of each sliced window
    skip_size : int
       the size of the window that should skipped before creating a new one
    Methods
    -------
    df_to_fixed_frames
       Creates a tuple of the frames containing the sequence of temporal series and the corresponding
       labels. Each frame is reduced to a vactor thanks to a PCA algorithm
    df_to_fixed_dct_frames
       Creates a tuple of the frames containing the DCT transform of the sequence of temporal series
       and the corresponding labels. Each frame is reduced to a vactor thanks to a PCA algorithm
    df_to_fixed_fft_frames
       Creates a tuple of the frames containing the FFT transform of the sequence of temporal series
       and the corresponding labels. Each frame is reduced to a vactor thanks to a PCA algorithm
    """

    def __init__(self, df):
        """
        Initialize the dataframe

        ...

        Parameters
        ----------
        df : dataframe
           the dataframe that contains all of the data
        """
        self.df = df

    def df_to_fixed_frames(self, frame_size, skip_size, transform_type='all'):
        """Creates a tuple of the frames containing the sequence of temporal series and the corresponding
        labels. Each frame is reduced to a vactor thanks to a PCA algorithm

        Parameters
        ----------
        frame_size : int
           the size of each sliced window
        skip_size : int
           the size of the window that should skipped before creating a new one

        Returns
        -------
        frames : array
            An array containing sub-arrays of each reduced frame of data
        labels : array
            An array containing labels corresponding to each sub-frame
        """
        # n_components = min(frame_size, 6)
        # frames = []
        acc_x_frames = []
        acc_y_frames = []
        acc_z_frames = []
        gyro_x_frames = []
        gyro_y_frames = []
        gyro_z_frames = []

        acc_x_frames_dct = []
        acc_y_frames_dct = []
        acc_z_frames_dct = []
        gyro_x_frames_dct = []
        gyro_y_frames_dct = []
        gyro_z_frames_dct = []

        acc_x_frames_fft = []
        acc_y_frames_fft = []
        acc_z_frames_fft = []
        gyro_x_frames_fft = []
        gyro_y_frames_fft = []
        gyro_z_frames_fft = []

        labels = []
        for i in range(0, len(self.df) - frame_size, skip_size):
            acc_x = self.df['Acc_x'].values[i:i + frame_size]
            acc_y = self.df['Acc_y'].values[i:i + frame_size]
            acc_z = self.df['Acc_z'].values[i:i + frame_size]

            gyro_x = self.df['Gyro_x'].values[i:i + frame_size]
            gyro_y = self.df['Gyro_y'].values[i:i + frame_size]
            gyro_z = self.df['Gyro_z'].values[i:i + frame_size]

            label = stats.mode(self.df['appui_leve'][i:i + frame_size])[0][0]

            if transform_type == 'all' or transform_type is None:
                acc_x_frames.append(np.asarray(acc_x))
                acc_y_frames.append(np.asarray(acc_y))
                acc_z_frames.append(np.asarray(acc_z))
                gyro_x_frames.append(np.asarray(gyro_x))
                gyro_y_frames.append(np.asarray(gyro_y))
                gyro_z_frames.append(np.asarray(gyro_z))

            elif transform_type == 'all' or transform_type == 'dct':
                acc_x_frames_dct.append(np.asarray(dct(acc_x)))
                acc_y_frames_dct.append(np.asarray(dct(acc_y)))
                acc_z_frames_dct.append(np.asarray(dct(acc_z)))
                gyro_x_frames_dct.append(np.asarray(dct(gyro_x)))
                gyro_y_frames_dct.append(np.asarray(dct(gyro_y)))
                gyro_z_frames_dct.append(np.asarray(dct(gyro_z)))

            elif transform_type == 'all' or transform_type == 'fft':
                acc_x_frames_fft.append(np.asarray(fft(acc_x).real))
                acc_y_frames_fft.append(np.asarray(fft(acc_y).real))
                acc_z_frames_fft.append(np.asarray(fft(acc_z).real))
                gyro_x_frames_fft.append(np.asarray(fft(gyro_x).real))
                gyro_y_frames_fft.append(np.asarray(fft(gyro_y).real))
                gyro_z_frames_fft.append(np.asarray(fft(gyro_z).real))

            labels.append(label)

        labels = np.asarray(labels)
        expand_dims_func = lambda x: np.expand_dims(np.asarray(x), axis=-1)
        if transform_type is None:
            frames = (acc_x_frames, acc_y_frames, acc_z_frames,
                      gyro_x_frames, gyro_y_frames, gyro_z_frames)
            frames = tuple(map(expand_dims_func, frames))
            return frames, labels
        elif transform_type == 'dct':
            dct_frames = (acc_x_frames_dct, acc_y_frames_dct, acc_z_frames_dct,
                          gyro_x_frames_dct, gyro_y_frames_dct, gyro_z_frames_dct)
            dct_frames = tuple(map(expand_dims_func, dct_frames))
            return dct_frames, labels
        elif transform_type == 'fft':
            fft_frames = (acc_x_frames_fft, acc_y_frames_fft, acc_z_frames_fft,
                          gyro_x_frames_fft, gyro_y_frames_fft, gyro_z_frames_fft)
            fft_frames = tuple(map(expand_dims_func, fft_frames))
            return fft_frames, labels
        elif transform_type == 'all':
            frames = (acc_x_frames, acc_y_frames, acc_z_frames, gyro_x_frames, gyro_y_frames, gyro_z_frames)
            frames = tuple(map(expand_dims_func, frames))
            dct_frames = (acc_x_frames_dct, acc_y_frames_dct, acc_z_frames_dct, gyro_x_frames_dct, gyro_y_frames_dct, gyro_z_frames_dct)
            dct_frames = tuple(map(expand_dims_func, dct_frames))
            fft_frames = (acc_x_frames_fft, acc_y_frames_fft, acc_z_frames_fft, gyro_x_frames_fft, gyro_y_frames_fft, gyro_z_frames_fft)
            fft_frames = tuple(map(expand_dims_func, fft_frames))
            all_frames = (frames, dct_frames, fft_frames)
            return all_frames, labels


class FormatData:
    """
    A class to generate a dictionnary containing training and test data as well as frames and
    dataframes that are sessential to recreate the original signal.

    ...

    Attributes
    ----------
    data_path : str
       the path to the csv file
    id_out : str
       the ID of the patient to use for test data
    train_df : dataframe
       the data frame created from the list patient IDs used fot the training set
    test_df : dataframe
       the data frame created from the patient ID used fot the test set
    train_var_frames : list
       the list of frames created from the training data
    test_var_frames : list
       the list of frames created from the test data
    transform : str
       the name of transform to use for data (None, dct or fft)
    window_size : int
       the size of each sliced window
    skip_size : int
       the size of the window that should skipped before creating a new one
    window_type : str
       the type of window to use, fixed or vary

    Methods
    -------
    loso_split
       Splits data into training and test data by Leaving One Subject One
    variate_window_df
       Creates a list of frames based on the target variable, each time
       the value of the target is swithing a new frame is created
    create_variate_frames
       Creates data matrices as well as the targets for both training and testing
    create_fixed_frames
       Creates data matrices as well as the targets for both training and testing
    """

    def __init__(self, data_path):
        """
        Initialize path to the data file, reads the file and initializes the data matrices

        ...

        Parameters
        ----------
        data_path : str
            The path leading to the data file
        """
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path)

        self.X_train = []
        self.X_dev = []
        self.X_test = []
        self.y_train = []
        self.y_dev = []
        self.y_test = []

    def loso_split(self, id_dev='P41', id_test='P42'):
        """Splits data into training and test data by Leaving One Subject One

        Parameters
        ----------
        id_out : str
            the ID of the patient to use for test data

        Returns
        -------
        train_df : dataframe
           the data frame created from the list patient IDs used fot the training set
        test_df : dataframe
           the data frame created from the patient ID used fot the test set
        """
        train_df = self.data[~self.data.id_pat.isin([id_dev, id_test])]

        dev_df = self.data[self.data.id_pat == id_dev]
        test_df = self.data[self.data.id_pat == id_test]
        return train_df, dev_df, test_df

    def create_fixed_frames(self, train_df, dev_df, test_df, window_size, skip_size, transform):
        """Creates a list of frames based on the chosen window, it
        corresponds to the most frequent label in each window

        Parameters
        ----------
        train_df : dataframe
           the data frame created from the list patient IDs used fot the training set
        test_df : dataframe
           the data frame created from the patient ID used fot the test set
        window_size : int
           the size of each sliced window
        skip_size : int
           the size of the window that should skipped before creating a new one
        transform : str
           the name of transform to use for data (None, dct or fft)

        Returns
        -------
        X_train : array
           an array that contains the training data
        X_test : array
           an array that contains the test data
        y_train : array
           an array that contains labels for the training data
        y_test : array
           an array that contains labels for the test data
        """
        frame_constructor = FixedFrameConstructor(train_df)
        X_train, y_train = frame_constructor.df_to_fixed_frames(window_size, skip_size, transform)
        frame_constructor = FixedFrameConstructor(dev_df)
        X_dev, y_dev = frame_constructor.df_to_fixed_frames(window_size, skip_size, transform)
        frame_constructor = FixedFrameConstructor(test_df)
        X_test, y_test = frame_constructor.df_to_fixed_frames(window_size, skip_size, transform)

        outputs = [y_train, y_dev, y_test]

        if transform == 'all':
            multi_inputs_none = (X_train[0], X_dev[0], X_test[0])
            multi_inputs_dct = (X_train[1], X_dev[1], X_test[1])
            multi_inputs_fft = (X_train[2], X_dev[2], X_test[2])

            uni_inputs_none = (np.hstack(X_train[0]), np.hstack(X_dev[0]), np.hstack(X_test[0]))
            uni_inputs_dct = (np.hstack(X_train[1]), np.hstack(X_dev[1]), np.hstack(X_test[1]))
            uni_inputs_fft = (np.hstack(X_train[2]), np.hstack(X_dev[2]), np.hstack(X_test[2]))

            data_none = (uni_inputs_none, multi_inputs_none, outputs)
            data_dct = (uni_inputs_dct, multi_inputs_dct, outputs)
            data_fft = (uni_inputs_fft, multi_inputs_fft, outputs)

            return data_none, data_dct, data_fft

        else:
            multi_inputs = (X_train, X_dev, X_test)
            uni_inputs = (X_train, np.hstack(X_dev), np.hstack(X_test))

            return uni_inputs, multi_inputs, outputs

    def __call__(self, transform_type, window_size=None, skip_size=None):
        """Applies transforms after class instance being called

        Parameters
        ----------
        window_type : str
           the type of window to use, fixed or vary
        window_size : int
           the size of each sliced window
        skip_size : int
           the size of the window that should skipped before creating a new one


        Returns
        -------
        my_frames : dict
           the dictionary that contains all of the combinaitions of generated data
        """
        train_df, dev_df, test_df = self.loso_split()
        # my_frames = {}
        assert (window_size is not None) or (
                skip_size is not None), "Arguments window_size and skip_size are missing"
        if transform_type == 'all':
            list_transforms = [None, 'dct', 'dct']
            temp_frames = self.create_fixed_frames(train_df, dev_df, test_df, window_size, skip_size, transform_type)
            my_frames = {transform: temp_frames[index] for (transform, index) in zip(list_transforms, [0, 1, 2])}
            my_frames['df_test'] = test_df
            #print(my_frames)
            #print("="*70)
            #print(my_frames[None])
            #print("="*70)
            #print(len(my_frames[None]))
            #print("="*70)
            #print(len(my_frames[None][0]))
            #print("="*70)
            #print(my_frames[None][0][0].shape)
        else:
            my_frames = {transform_type: self.create_fixed_frames(train_df, dev_df, test_df, window_size, skip_size,
                                                                  transform_type), 'df_test': test_df}
        return my_frames
