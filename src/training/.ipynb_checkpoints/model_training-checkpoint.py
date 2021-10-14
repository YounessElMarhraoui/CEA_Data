import os

import numpy as np
import pandas as pd

from .neural_models import CNNModel, CNNModel_comp, MultiLSTMModel, UniModalLSTMModel, MultiCNNModel, \
    CNN_two_channels_Model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class modelTrain:
    """
    A class to create, train and evaluate the DL model

    ...

    Attributes
    ----------
    frames_dict : dict
       the dictionary that contains all of the combinaitions of generated data
    window_type : str
       the type of window to use, fixed or vary
    transform_type : str
       the name of transform to use for data (None, dct or fft)

    Methods
    -------
    fit_models
       Fits the models on the training data
    evaluate_models
       Evaualtes the predictions made by the models by reformatting them into their initial length
    """

    def __init__(self, frames_dict, window_size, transform_type, model_store, data_store):
        """
        Initialize dictionary of the different types of frames, the type of
        windows and the data transformation type as well as the classifier
        to train

        ...

        Parameters
        ----------
        frames_dict : dict
           the dictionary that contains all of the combinaitions of generated data
        window_type : str
           the type of window to use, fixed or vary
        transform_type : str
           the name of transform to use for data (None, dct or fft)
        """
        self.model_store = model_store
        self.data_store = data_store
        self.frames_dict = frames_dict
        self.window_size = window_size
        self.transform_type = transform_type
        self.uni_input, self.multi_input, self.outputs = self.frames_dict[self.transform_type]
        #print(len(self.uni_input))
        #print(self.uni_input[0].shape)
        #print(len(self.multi_input))
        #print(self.multi_input[0].shape)
        self.X_train, self.y_train, self.X_dev, self.y_dev, self.X_test, self.y_test = None, None, None, None, None, None
        self.test_df = {}
        self.test_var_frames = {}
        self.test_df = self.frames_dict['df_test']

        self.list_models = {'cnn_model': CNNModel(),
                            'cnn_model_comp': CNNModel_comp(),
                            'uni_lstm_model': UniModalLSTMModel(),
                            'multi_lstm_model': MultiLSTMModel(),
                            'multi_cnn_model': MultiCNNModel(),
                            'cnn_two_channels_model': CNN_two_channels_Model()}

    def train_model(self, model, model_name, n_epochs=20, bs=800):
        if model_name in ['multi_lstm_model', 'multi_cnn_model']:
            self.X_train = self.multi_input[0]
            self.X_dev = self.multi_input[1]
            self.X_test = self.multi_input[2]
        elif model_name == 'cnn_two_channels_model':
            self.X_train = self.uni_input[0].reshape(-1, 2, 3, self.window_size, 1)
            self.X_dev = self.uni_input[1].reshape(-1, 2, 3, self.window_size, 1)
            self.X_test = self.uni_input[2].reshape(-1, 2, 3, self.window_size, 1)
        else:
            self.X_train = self.uni_input[0].reshape(-1, 6, self.window_size,1)
            self.X_dev = self.uni_input[1].reshape(-1, 6, self.window_size, 1)
            self.X_test = self.uni_input[2].reshape(-1, 6, self.window_size, 1)

        self.y_train = self.outputs[0]
        self.y_dev = self.outputs[1]
        self.y_test = self.outputs[2]

        # print(self.X_train)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(self.X_train, self.y_train, validation_data=(self.X_dev, self.y_dev),
                  epochs=n_epochs, batch_size=bs, verbose=2)
        model.save(self.model_store + '/' + model_name + '_ws_{}_transform_{}'.format(
            self.window_size, self.transform_type))

    def predict(self, model):
        y_pred = model.predict(self.X_test)
        model.evaluate(self.X_test, self.y_test)
        # print("Xtest shape: ",self.X_test.shape)
        # print("ytest shape: ",self.y_test.shape)
        # print(y_pred.shape)
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        y_real_pred = [[ele] * self.window_size for ele in y_pred]
        y_real_pred = np.asarray(y_real_pred).flatten()
        y_real_test = self.test_df.appui_leve.values[:len(y_real_pred)]
        # print("Y test shape: ", y_real_test.shape)
        # print("Y pred shape: ", y_real_pred.shape)
        print("Accuracy score: ", accuracy_score(y_real_test, y_real_pred))
        print("Confusion matrix: \n", confusion_matrix(y_real_test, y_real_pred))
        print("Classification report: \n", classification_report(y_real_test, y_real_pred))
        return y_real_test, y_real_pred

    def store_predictions(self, model, model_name):
        y_real, y_pred = self.predict(model)
        predictions = pd.DataFrame({'real_values': y_real, 'predictions': y_pred})
        predictions.to_csv(
            self.data_store + '/' + model_name + '_ws_{}_transform_{}.csv'.format(self.window_size, self.transform_type))

    def evaluate(self, y_real, y_pred):
        print("Accuracy score: ", accuracy_score(y_real, y_pred))
        print("Confusion matrix: \n", confusion_matrix(y_real, y_pred))
        print("Classification report: \n", classification_report(y_real, y_pred))

        plt.figure(figsize=(20, 5))
        plt.plot(y_real, label='Real')
        plt.plot(y_pred, label='Predictions')
        plt.legend()
        plt.figure(figsize=(20, 5))
        plt.plot(y_real)
        plt.figure(figsize=(20, 5))
        plt.plot(y_pred)
        plt.show()

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_real, name='Real values'))
        fig.add_trace(go.Scatter(y=y_pred, name='Predicted values'))
        pio.show(fig)

    def evaluate_models(self):
        """Evaualtes the predictions made by the models by reformatting them into their initial length
        """
        cnn_score = self.cnn_model.evaluate(self.X_test, self.y_test)
        y_pred_cnn = self.cnn_model.predict(self.X_test)
        y_pred_cnn = np.where(y_pred_cnn >= 0.5, 1, 0)
        y_real_test_pred = [[ele] * self.window_size for ele in y_pred_cnn]
        y_real_test_pred = np.asarray(y_real_test_pred).flatten()
        y_real_test = self.test_df.appui_leve.values[:len(y_real_test_pred)]
        print(
            """
            ##################################
            #            CNN model           #
            ##################################
            """
        )
        print("########## All test data ##########")
        self.evaluate(y_real_test_pred, y_real_test)

        print("########## Left/Right data ##########")
        self.test_df['preds'] = np.append(y_real_test_pred, np.zeros(self.test_df.shape[0] - len(y_real_test_pred)))
        test_df_g = self.test_df[self.test_df.foot_type == 0]
        test_df_d = self.test_df[self.test_df.foot_type == 1]
        self.evaluate(test_df_g.appui_leve, test_df_g.preds)
        self.evaluate(test_df_d.appui_leve, test_df_d.preds)

        print("########## Per activity data ##########")
        for activity in test_df_g.activity.unique():
            print("#### Activité: {} ####".format(activity))
            temp_df_g = test_df_g[test_df_g.activity == activity]
            temp_df_d = test_df_d[test_df_d.activity == activity]
            self.evaluate(temp_df_g.appui_leve, temp_df_g.preds)
            self.evaluate(temp_df_d.appui_leve, temp_df_d.preds)

    def pipeline(self, model, model_name):
        self.train_model(model, model_name)
        self.store_predictions(model, model_name)

    def __call__(self):
        for model_name, model in self.list_models.items():
            if model_name + '_ws_{}_transform_{}'.format(self.window_size, self.transform_type) in os.listdir(self.model_store):
                print(model_name + '_ws_{}_transform_{}'.format(self.window_size, self.transform_type) + " is already trained!!")
                continue
            else:
                print(model_name + ' is training!')
                print("#" * 60)
                self.pipeline(model, model_name)
        self.test_df.to_csv(self.data_store + '/test_df.csv')
        # self.evaluate_models()
