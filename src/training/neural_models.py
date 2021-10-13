import tensorflow as tf


class CNNModel(tf.keras.Model):
    """
    A class to create the arcitecture of the DNN model

    ...

    Attributes
    ----------
    inputs : array
       the array of inputs that the model would train on
    """

    def __init__(self):
        """
        Initialize the layers of the model
        """
        super(CNNModel, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=3, activation=tf.nn.relu)
        self.conv2 = tf.keras.layers.Conv2D(32, kernel_size=3, activation=tf.nn.relu)
        self.flt = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs):
        """Forwad propagates the inputs into the model

        Parameters
        ----------
        inputs : array
           the array of inputs that the model would train on

        Returns
        -------
        x : tensor
            the output of the model
        """
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flt(x)
        x = self.fc1(x)
        return x


class CNNModel_comp(tf.keras.Model):
    """
    A class to create the arcitecture of the DNN model

    ...

    Attributes
    ----------
    inputs : array
       the array of inputs that the model would train on
    """

    def __init__(self):
        """
        Initialize the layers of the model
        """
        super(CNNModel_comp, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.ReLU()

        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.act3 = tf.keras.layers.ReLU()

        self.gap = tf.keras.layers.GlobalAveragePooling2D()

        self.flt = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs):
        """Forwad propagates the inputs into the model

        Parameters
        ----------
        inputs : array
           the array of inputs that the model would train on

        Returns
        -------
        x : tensor
            the output of the model
        """
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.gap(x)
        x = self.flt(x)
        x = self.fc1(x)
        return x


class MultiLSTMModel(tf.keras.Model):
    """
    A class to create the arcitecture of the DNN model

    ...

    Attributes
    ----------
    inputs : array
       the array of inputs that the model would train on
    """

    def __init__(self):
        """
        Initialize the layers of the model
        """
        super(MultiLSTMModel, self).__init__()

        self.lstm1_acc_x = tf.keras.layers.LSTM(8, return_sequences=True, activation=tf.nn.relu)
        self.lstm1_acc_y = tf.keras.layers.LSTM(8, return_sequences=True, activation=tf.nn.relu)
        self.lstm1_acc_z = tf.keras.layers.LSTM(8, return_sequences=True, activation=tf.nn.relu)

        self.lstm1_gyro_x = tf.keras.layers.LSTM(8, return_sequences=True, activation=tf.nn.relu)
        self.lstm1_gyro_y = tf.keras.layers.LSTM(8, return_sequences=True, activation=tf.nn.relu)
        self.lstm1_gyro_z = tf.keras.layers.LSTM(8, return_sequences=True, activation=tf.nn.relu)

        self.lstm2_acc_x = tf.keras.layers.LSTM(8, activation=tf.nn.relu)
        self.lstm2_acc_y = tf.keras.layers.LSTM(8, activation=tf.nn.relu)
        self.lstm2_acc_z = tf.keras.layers.LSTM(8, activation=tf.nn.relu)

        self.lstm2_gyro_x = tf.keras.layers.LSTM(8, activation=tf.nn.relu)
        self.lstm2_gyro_y = tf.keras.layers.LSTM(8, activation=tf.nn.relu)
        self.lstm2_gyro_z = tf.keras.layers.LSTM(8, activation=tf.nn.relu)

        self.flt_acc_x = tf.keras.layers.Flatten()
        self.flt_acc_y = tf.keras.layers.Flatten()
        self.flt_acc_z = tf.keras.layers.Flatten()
        self.flt_gyro_x = tf.keras.layers.Flatten()
        self.flt_gyro_y = tf.keras.layers.Flatten()
        self.flt_gyro_z = tf.keras.layers.Flatten()

        self.concat = tf.keras.layers.Concatenate()
        self.fc = tf.keras.layers.Dense(100, activation=tf.nn.relu)
        self.fc_1 = tf.keras.layers.Dense(50, activation=tf.nn.relu)

        self.fc1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs):
        """Forwad propagates the inputs into the model

        Parameters
        ----------
        inputs : array
           the array of inputs that the model would train on

        Returns
        -------
        x : tensor
            the output of the model

        in_acc_x = tf.reshape(in_acc_x, [-1, in_acc_x.shape[0], in_acc_x.shape[1]])
        in_acc_x.set_shape([None, in_acc_x.shape[1], in_acc_x.shape[2]])

        in_acc_y = tf.reshape(in_acc_y, [-1, in_acc_y.shape[0], in_acc_y.shape[1]])
        in_acc_y.set_shape([None, in_acc_y.shape[1], in_acc_y.shape[2]])

        in_acc_z = tf.reshape(in_acc_z, [-1, in_acc_z.shape[0], in_acc_z.shape[1]])
        in_acc_z.set_shape([None, in_acc_z.shape[1], in_acc_z.shape[2]])

        in_gyro_x = tf.reshape(in_gyro_x, [-1, in_gyro_x.shape[0], in_gyro_x.shape[1]])
        in_gyro_x.set_shape([None, in_gyro_x.shape[1], in_gyro_x.shape[2]])

        in_gyro_y = tf.reshape(in_gyro_y, [-1, in_gyro_y.shape[0], in_gyro_y.shape[1]])
        in_gyro_y.set_shape([None, in_gyro_y.shape[1], in_gyro_y.shape[2]])

        in_gyro_z = tf.reshape(in_gyro_z, [-1, in_gyro_z.shape[0], in_gyro_z.shape[1]])
        in_gyro_z.set_shape([None, in_gyro_z.shape[1], in_gyro_z.shape[2]])
        """
        in_acc_x = inputs[0]
        in_acc_y = inputs[1]
        in_acc_z = inputs[2]
        in_gyro_x = inputs[3]
        in_gyro_y = inputs[4]
        in_gyro_z = inputs[5]

        # In here reshaping

        in_acc_x = self.lstm1_acc_x(in_acc_x)
        in_acc_y = self.lstm1_acc_y(in_acc_y)
        in_acc_z = self.lstm1_acc_z(in_acc_z)
        in_gyro_x = self.lstm1_gyro_x(in_gyro_x)
        in_gyro_y = self.lstm1_gyro_y(in_gyro_y)
        in_gyro_z = self.lstm1_gyro_z(in_gyro_z)

        in_acc_x = self.lstm2_acc_x(in_acc_x)
        in_acc_y = self.lstm2_acc_y(in_acc_y)
        in_acc_z = self.lstm2_acc_z(in_acc_z)
        in_gyro_x = self.lstm2_gyro_x(in_gyro_x)
        in_gyro_y = self.lstm2_gyro_y(in_gyro_y)
        in_gyro_z = self.lstm2_gyro_z(in_gyro_z)

        in_acc_x = self.flt_acc_x(in_acc_x)
        in_acc_y = self.flt_acc_y(in_acc_y)
        in_acc_z = self.flt_acc_z(in_acc_z)
        in_gyro_x = self.flt_gyro_x(in_gyro_x)
        in_gyro_y = self.flt_gyro_y(in_gyro_y)
        in_gyro_z = self.flt_gyro_z(in_gyro_z)

        x = self.concat([in_acc_x, in_acc_y, in_acc_z, in_gyro_x, in_gyro_y, in_gyro_z])

        x = self.fc(x)
        x = self.fc_1(x)
        x = self.fc1(x)
        return x


class UniModalLSTMModel(tf.keras.Model):
    """
    A class to create the arcitecture of the DNN model

    ...

    Attributes
    ----------
    inputs : array
       the array of inputs that the model would train on
    """

    def __init__(self):
        """
        Initialize the layers of the model
        """
        super(UniModalLSTMModel, self).__init__()

        self.lstm1 = tf.keras.layers.LSTM(4, return_sequences=True, activation=tf.nn.relu)
        self.lstm2 = tf.keras.layers.LSTM(4, activation=tf.nn.relu)

        self.flt = tf.keras.layers.Flatten()

        self.fc1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs):
        """Forwad propagates the inputs into the model

        Parameters
        ----------
        inputs : array
           the array of inputs that the model would train on

        Returns
        -------
        x : tensor
            the output of the model
        """
        inputs = tf.squeeze(inputs, axis=-1)
        x = self.lstm1(inputs)

        x = self.lstm2(x)

        x = self.flt(x)

        x = self.fc1(x)
        return x


class MultiCNNModel(tf.keras.Model):
    """
    A class to create the arcitecture of the DNN model

    ...

    Attributes
    ----------
    inputs : array
       the array of inputs that the model would train on
    """

    def __init__(self):
        """
        Initialize the layers of the model
        """
        super(MultiCNNModel, self).__init__()

        self.cnn1_acc_x = tf.keras.layers.Conv1D(filters=8, kernel_size=3, padding="same", activation='selu')
        self.cnn1_acc_y = tf.keras.layers.Conv1D(filters=8, kernel_size=3, padding="same", activation='selu')
        self.cnn1_acc_z = tf.keras.layers.Conv1D(filters=8, kernel_size=3, padding="same", activation='selu')

        self.cnn1_gyro_x = tf.keras.layers.Conv1D(filters=8, kernel_size=3, padding="same", activation='selu')
        self.cnn1_gyro_y = tf.keras.layers.Conv1D(filters=8, kernel_size=3, padding="same", activation='selu')
        self.cnn1_gyro_z = tf.keras.layers.Conv1D(filters=8, kernel_size=3, padding="same", activation='selu')

        self.cnn2_acc_x = tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding="same", activation='selu')
        self.cnn2_acc_y = tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding="same", activation='selu')
        self.cnn2_acc_z = tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding="same", activation='selu')

        self.cnn2_gyro_x = tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding="same", activation='selu')
        self.cnn2_gyro_y = tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding="same", activation='selu')
        self.cnn2_gyro_z = tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding="same", activation='selu')

        self.max1_acc_x = tf.keras.layers.MaxPooling1D(2)
        self.max1_acc_y = tf.keras.layers.MaxPooling1D(2)
        self.max1_acc_z = tf.keras.layers.MaxPooling1D(2)

        self.max1_gyro_x = tf.keras.layers.MaxPooling1D(2)
        self.max1_gyro_y = tf.keras.layers.MaxPooling1D(2)
        self.max1_gyro_z = tf.keras.layers.MaxPooling1D(2)

        self.cnn3_acc_x = tf.keras.layers.Conv1D(filters=8, kernel_size=3, padding="same", activation='selu')
        self.cnn3_acc_y = tf.keras.layers.Conv1D(filters=8, kernel_size=3, padding="same", activation='selu')
        self.cnn3_acc_z = tf.keras.layers.Conv1D(filters=8, kernel_size=3, padding="same", activation='selu')

        self.cnn3_gyro_x = tf.keras.layers.Conv1D(filters=8, kernel_size=3, padding="same", activation='selu')
        self.cnn3_gyro_y = tf.keras.layers.Conv1D(filters=8, kernel_size=3, padding="same", activation='selu')
        self.cnn3_gyro_z = tf.keras.layers.Conv1D(filters=8, kernel_size=3, padding="same", activation='selu')

        self.cnn4_acc_x = tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding="same", activation='selu')
        self.cnn4_acc_y = tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding="same", activation='selu')
        self.cnn4_acc_z = tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding="same", activation='selu')

        self.cnn4_gyro_x = tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding="same", activation='selu')
        self.cnn4_gyro_y = tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding="same", activation='selu')
        self.cnn4_gyro_z = tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding="same", activation='selu')

        self.max2_acc_x = tf.keras.layers.MaxPooling1D(2)
        self.max2_acc_y = tf.keras.layers.MaxPooling1D(2)
        self.max2_acc_z = tf.keras.layers.MaxPooling1D(2)

        self.max2_gyro_x = tf.keras.layers.MaxPooling1D(2)
        self.max2_gyro_y = tf.keras.layers.MaxPooling1D(2)
        self.max2_gyro_z = tf.keras.layers.MaxPooling1D(2)

        self.flt_acc_x = tf.keras.layers.Flatten()
        self.flt_acc_y = tf.keras.layers.Flatten()
        self.flt_acc_z = tf.keras.layers.Flatten()

        self.flt_gyro_x = tf.keras.layers.Flatten()
        self.flt_gyro_y = tf.keras.layers.Flatten()
        self.flt_gyro_z = tf.keras.layers.Flatten()

        self.fc_acc_x = tf.keras.layers.Dense(50, activation='elu')
        self.fc_acc_y = tf.keras.layers.Dense(50, activation='elu')
        self.fc_acc_z = tf.keras.layers.Dense(50, activation='elu')

        self.fc_gyro_x = tf.keras.layers.Dense(50, activation='elu')
        self.fc_gyro_y = tf.keras.layers.Dense(50, activation='elu')
        self.fc_gyro_z = tf.keras.layers.Dense(50, activation='elu')

        self.concat = tf.keras.layers.Concatenate()

        self.drp1 = tf.keras.layers.Dropout(0.5)
        self.fc_1 = tf.keras.layers.Dense(100, activation='selu')
        self.drp2 = tf.keras.layers.Dropout(0.5)
        self.fc_2 = tf.keras.layers.Dense(20, activation='selu')
        self.drp3 = tf.keras.layers.Dropout(0.5)

        self.fc_dec = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs):
        """Forwad propagates the inputs into the model

        Parameters
        ----------
        inputs : array
           the array of inputs that the model would train on

        Returns
        -------
        x : tensor
            the output of the model
        """
        in_acc_x = inputs[0]
        in_acc_y = inputs[1]
        in_acc_z = inputs[2]
        in_gyro_x = inputs[3]
        in_gyro_y = inputs[4]
        in_gyro_z = inputs[5]

        # in_acc_x = tf.squeeze(in_acc_x, axis=-1)
        # print(in_acc_x.shape)
        # in_acc_y = tf.squeeze(in_acc_y, axis=-1)
        # in_acc_z = tf.squeeze(in_acc_z, axis=-1)
        # in_gyro_x = tf.squeeze(in_gyro_x, axis=-1)
        # in_gyro_y = tf.squeeze(in_gyro_y, axis=-1)
        # in_gyro_z = tf.squeeze(in_gyro_z, axis=-1)

        in_acc_x = self.cnn1_acc_x(in_acc_x)
        in_acc_y = self.cnn1_acc_y(in_acc_y)
        in_acc_z = self.cnn1_acc_z(in_acc_z)
        in_gyro_x = self.cnn1_gyro_x(in_gyro_x)
        in_gyro_y = self.cnn1_gyro_y(in_gyro_y)
        in_gyro_z = self.cnn1_gyro_z(in_gyro_z)

        in_acc_x = self.cnn2_acc_x(in_acc_x)
        in_acc_y = self.cnn2_acc_y(in_acc_y)
        in_acc_z = self.cnn2_acc_z(in_acc_z)
        in_gyro_x = self.cnn2_gyro_x(in_gyro_x)
        in_gyro_y = self.cnn2_gyro_y(in_gyro_y)
        in_gyro_z = self.cnn2_gyro_z(in_gyro_z)

        in_acc_x = self.max1_acc_x(in_acc_x)
        in_acc_y = self.max1_acc_y(in_acc_y)
        in_acc_z = self.max1_acc_z(in_acc_z)
        in_gyro_x = self.max1_gyro_x(in_gyro_x)
        in_gyro_y = self.max1_gyro_y(in_gyro_y)
        in_gyro_z = self.max1_gyro_z(in_gyro_z)

        in_acc_x = self.cnn3_acc_x(in_acc_x)
        in_acc_y = self.cnn3_acc_y(in_acc_y)
        in_acc_z = self.cnn3_acc_z(in_acc_z)
        in_gyro_x = self.cnn3_gyro_x(in_gyro_x)
        in_gyro_y = self.cnn3_gyro_y(in_gyro_y)
        in_gyro_z = self.cnn3_gyro_z(in_gyro_z)

        in_acc_x = self.cnn4_acc_x(in_acc_x)
        in_acc_y = self.cnn4_acc_y(in_acc_y)
        in_acc_z = self.cnn4_acc_z(in_acc_z)
        in_gyro_x = self.cnn4_gyro_x(in_gyro_x)
        in_gyro_y = self.cnn4_gyro_y(in_gyro_y)
        in_gyro_z = self.cnn4_gyro_z(in_gyro_z)

        in_acc_x = self.max2_acc_x(in_acc_x)
        in_acc_y = self.max2_acc_y(in_acc_y)
        in_acc_z = self.max2_acc_z(in_acc_z)
        in_gyro_x = self.max2_gyro_x(in_gyro_x)
        in_gyro_y = self.max2_gyro_y(in_gyro_y)
        in_gyro_z = self.max2_gyro_z(in_gyro_z)

        in_acc_x = self.fc_acc_x(in_acc_x)
        in_acc_y = self.fc_acc_y(in_acc_y)
        in_acc_z = self.fc_acc_z(in_acc_z)
        in_gyro_x = self.fc_gyro_x(in_gyro_x)
        in_gyro_y = self.fc_gyro_y(in_gyro_y)
        in_gyro_z = self.fc_gyro_z(in_gyro_z)

        in_acc_x = self.flt_acc_x(in_acc_x)
        in_acc_y = self.flt_acc_y(in_acc_y)
        in_acc_z = self.flt_acc_z(in_acc_z)
        in_gyro_x = self.flt_gyro_x(in_gyro_x)
        in_gyro_y = self.flt_gyro_y(in_gyro_y)
        in_gyro_z = self.flt_gyro_z(in_gyro_z)

        x = self.concat([in_acc_x, in_acc_y, in_acc_z, in_gyro_x, in_gyro_y, in_gyro_z])

        x = self.drp1(x)
        x = self.fc_1(x)
        x = self.drp2(x)
        x = self.fc_2(x)
        x = self.drp3(x)

        x = self.fc_dec(x)
        return x


class CNN_two_channels_Model(tf.keras.Model):
    """
    A class to create the arcitecture of the DNN model

    ...

    Attributes
    ----------
    inputs : array
       the array of inputs that the model would train on
    """

    def __init__(self):
        """
        Initialize the layers of the model
        """
        super(CNN_two_channels_Model, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.ReLU()

        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.act3 = tf.keras.layers.ReLU()

        self.gap = tf.keras.layers.GlobalAveragePooling3D()

        self.flt = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs):
        """Forwad propagates the inputs into the model

        Parameters
        ----------
        inputs : array
           the array of inputs that the model would train on

        Returns
        -------
        x : tensor
            the output of the model
        in_acc_x = inputs[0][0]
        in_acc_y = inputs[0][1]
        in_acc_z = inputs[0][2]
        in_gyro_x = inputs[0][3]
        in_gyro_y = inputs[0][4]
        in_gyro_z = inputs[0][5]

        in_acc = K.stack([in_acc_x, in_acc_y, in_acc_z])
        in_gyro = K.stack([in_gyro_x, in_gyro_y, in_gyro_z])
        in_all = K.stack([in_acc, in_gyro])
        in_all = tf.reshape(in_all, [-1, in_all.shape[0], in_all.shape[1], in_all.shape[2]])
        """

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)

        x = self.gap(x)
        x = self.flt(x)
        x = self.fc1(x)
        return x
