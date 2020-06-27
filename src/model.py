'''
Created on 19 may. 2020

@author: Luis
'''
import tensorflow as tf
import numpy.random as rd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.layers import Dense, Dropout, LSTM,  BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, CSVLogger, EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
import tensorflow.keras.backend as K
import numpy as np
from src.historicData import normalize, getLstm
def train_model(df):
    print('training of dataset started')
    #normalization
    columns=df.columns[8:-6]
    dfTrain =normalize(df)

    #Creación dataset LSTM 
    dfLstmTrain = getLstm(dfTrain);

    #Filtramos las columnas
    dfColsTrain = dfLstmTrain[columns]
    res = [];
    y=[]
    rr=True

    for val in dfColsTrain.index.get_level_values(0).unique():
        for val2 in dfColsTrain.loc[val].index.get_level_values(0).unique():
            res.append(dfColsTrain.loc[val,val2].values);
            y.append(dfLstmTrain.loc[val,val2,val2]['target4_8'])
    resnp = np.asarray(res)
    ynp = np.asarray(y)

    #resnp=np.delete(resnp,slice(0,10),1)
    from sklearn.utils import shuffle
    X_total,y_total = shuffle(resnp,ynp);

    #X_total=tf.keras.utils.normalize(X_total,axis=1,order=1)
    x_train=X_total[:-100];
    y_train=tf.keras.utils.to_categorical(
        y_total[:-100], num_classes=3, dtype='int16'
    )
    x_test=X_total[-100:];
    y_test=tf.keras.utils.to_categorical(
        y_total[-100:], num_classes=3, dtype='int16'
    )
    model2 = Sequential()


    model2.add(LSTM(128, input_shape=(x_train.shape[1:]),return_sequences=False))
    model2.add(Dropout(0.1))
    model2.add(BatchNormalization()) 
    model2.add(Dense(3, activation='softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    EPOCHS = 70  # how many passes through our data
    BATCH_SIZE = 60
    # Compile model

    def f1(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # Replace class_id_preds with class_id_true for recall here
        accuracy_mask = K.cast(K.equal(class_id_preds, 1), 'int32')
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc

    def f2(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # Replace class_id_preds with class_id_true for recall here
        accuracy_mask = K.cast(K.equal(class_id_preds, 2), 'int32')
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc
    def f0(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # Replace class_id_preds with class_id_true for recall here
        accuracy_mask = K.cast(K.equal(class_id_preds, 0), 'int32')
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc

    class_weights = class_weight.compute_class_weight('balanced',
                                             [0,1,2],
                                             y_total[:-1])


    import tensorflow.keras.backend as K
    from sklearn.metrics import classification_report



    model2.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy',f1,f0,f2]
    )

   
    #print(class_w)
    #class_w = {0:5.5,1:1.0,2:4.5}
    model2.summary();
    class_weights = class_weight.compute_class_weight('balanced',
                                         [0,1,2],
                                         y_total[:-1])
    class_w={0:class_weights[0]*(1/class_weights[1]),1:1.0,2:class_weights[2]*(1/class_weights[1])}
    history = model2.fit(
        x_train,y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        class_weight=class_w,
        verbose=0
    )

    hist = history.history;
    return hist