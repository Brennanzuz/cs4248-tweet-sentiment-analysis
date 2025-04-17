import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input, Dense, Dropout, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint


from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator
import sklearn.metrics as metrics

from typing import List, Dict, Any, Tuple
import math


class ClassificationModel(BaseEstimator):
    def __init__(*kwargs):
        raise NotImplementedError
    

class TfModel(ClassificationModel):
    def __init__(self, layer_list: List[Tuple[str, Any]], batch_size:int, dropout_rate:float=0.2):
        curr_layer = None
        self.dropout_rate = dropout_rate
        self.layer_list = layer_list
        for l, conf in layer_list:
            match l:
                case "input":
                    curr_layer = Input(**conf)
                    input_ts = curr_layer
                case "dense":
                    curr_layer = Dense(**conf)(curr_layer)
                case "dropout":
                    curr_layer = Dropout(dropout_rate)(curr_layer)
                case "output":
                    curr_layer = Dense(**conf)(curr_layer)
                    output_ts = curr_layer
                case "lstm":
                    curr_layer = LSTM(**conf)(curr_layer)

        self.model = Model(inputs=input_ts, outputs=output_ts)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        self.model.compile(loss="categorical_crossentropy",
                        optimizer=tf.optimizers.Adam(),
                        metrics=["accuracy"])
        self.lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-4 * math.exp(-0.1 * x))
        self.batch_size = batch_size
        
    # Removed to fit the Scikit evaluator API  
    # def fit(self, x_train, x_valid, y_train, y_valid, verbose=True): 
    #     history = self.model.fit(x_train, y_train,                   
    #                         epochs=10,
    #                         validation_data=(x_valid, y_valid),
    #                         batch_size=self.batch_size,
    #                         verbose=verbose,
    #                         callbacks=[self.lr_scheduler, self.early_stopping])
    #     return history
    def fit(self, x_train, y_train, verbose=True):
        self.model.fit(x_train, y_train,                   
                        epochs=10,
                        batch_size=self.batch_size,
                        verbose=verbose,
                        callbacks=[self.lr_scheduler])
        return self
    
    def save_model(self):
        # Save the trained model
        self.model.save('model.keras')
        return
    
    def predict(self, x_test):
        probs = self.model.predict_on_batch(x_test)
        preds_index = probs.argmax(axis=1)
        preds_onehot = np.zeros(probs.shape)
        for i in range(preds_index.shape[0]):
            preds_onehot[i][preds_index[i]] = 1
        # print(preds_onehot)
        return preds_onehot

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        # print(self.layer_list)
        for l in self.layer_list:
            # print(l)
            match l[0]:
                case "input":
                    curr_layer = Input(**l[1])
                    input_ts = curr_layer
                case "dense":
                    curr_layer = Dense(**l[1])(curr_layer)
                case "dropout":
                    curr_layer = Dropout(self.dropout_rate)(curr_layer)
                case "output":
                    curr_layer = Dense(**l[1])(curr_layer)
                    output_ts = curr_layer
                case "lstm":
                    curr_layer = LSTM(**l[1])(curr_layer)

        self.model = Model(inputs=input_ts, outputs=output_ts)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        self.model.compile(loss="categorical_crossentropy",
                        optimizer=tf.optimizers.Adam(),
                        metrics=["accuracy"])
        return self

# class KnnModel(ClassificationModel):
#     def __init__(self, conf):
#         self.model = KNeighborsClassifier(n_neighbors=3, **conf)
    
#     def fit(self, x_train, x_valid, y_train, y_valid, verbose=True):
#         self.model.fit(x_train, y_train)
#         preds = self.model.predict(x_valid)
#         if verbose:
#             print(metrics.classification_report(y_valid, preds))
#         return preds

#     def predict(self, x_test):
#         self.model.predict(x_test)