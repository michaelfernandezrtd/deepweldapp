from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, concatenate, GlobalMaxPool2D
from tensorflow.keras.models import Model
from keras.layers import MaxPooling2D, Conv2D, InputLayer, Input
from keras import losses, callbacks, regularizers
from tensorflow.python.client import device_lib
from keras.utils import multi_gpu_model
import tensorflow as tf
import keras
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def distortion_model_hybrid(X_train, Y_train, X_val, Y_val, params):

        print("In", len(X_train))
        #nb_conv = 4
        #nb_pool = 2
        batch_size = 64
        nb_epoch = 1000

        nb_filters = params['nb_filters']
        nb_conv = params['nb_conv']
        nb_pool = params['nb_pool']
        nb_layer = params['nb_layer']
        dropout = params['dropout']
        hidden = params['nb_hidden']

        nb_classes = Y_train.shape[1]
        input_shape = (8, 8)

        input_pattern = Input(shape=input_shape, name='input1')
        input_geo = Input(shape=(X_train[1].shape[1],), name='input2')

        # build the rest of the network
        model = Conv2D(nb_filters, (nb_conv, nb_conv),
                       padding='valid',
                       input_shape=input_shape, name='conv2d_0')(input_pattern)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = Dropout(dropout)(model)
        model = MaxPooling2D(pool_size=(nb_pool, nb_pool))(model)
        model = BatchNormalization()(model)
        # model = Activation('relu')(model)

        for i in range(1, nb_layer):
            model = Conv2D(nb_filters, nb_conv, nb_conv, name='conv2d_' + str(i))(model)
            model = BatchNormalization()(model)
            model = Activation('relu')(model)
            model = Dropout(dropout)(model)
            model = MaxPooling2D(pool_size=(nb_pool, nb_pool))(model)
            model = BatchNormalization()(model)
        # model = Activation('relu')(model)

        model = GlobalMaxPool2D()(model)
        # model = Flatten()(model)
        model = concatenate([model, input_geo])
            # print(model.output.shape, geometrical_input.shape)

        model = Dense(hidden, activation='relu')(model)
        model = Activation('relu')(model)
        # model.add(Activation('sigmoid'))
        model = Dropout(dropout)(model)
        out = Dense(nb_classes, name='dense_output', activation='linear')(model)

        if type(X_train.type) is list:
            model_final = Model(inputs=[input_pattern, input_geo], outputs=out)
        # model.add(Activation('sigmoid'))
        else:
            model_final = Model(inputs=[input_pattern], outputs=out)
        model_final.summary()

        model_final.compile(loss='mean_squared_error', optimizer='adam')
        history = model_final.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                  verbose=0, validation_data=(X_val, Y_val), shuffle=True, callbacks=[tbCallBck, earlyStopping])

        return history


def distortion_model_functional(X_train, Y_train, X_val, Y_val, params):
    print("In", len(X_train))
    # nb_conv = 4
    # nb_pool = 2
    batch_size = 64
    nb_epoch = 1000
    opt = keras.optimizers.Adam()

    nb_filters = params['nb_filters']
    nb_conv = params['nb_conv']
    nb_pool = params['nb_pool']
    nb_layer = params['nb_layer']
    dropout = params['dropout']
    hidden = params['nb_hidden']

    nb_classes = Y_train.shape[1]
    input_shape = (8, 8, 1)

    input_pattern = Input(shape=input_shape, name='input1')

    # build the rest of the network
    model = Conv2D(nb_filters, (nb_conv, nb_conv),
                   padding='valid',
                   input_shape=input_shape, name='conv2d_0')(input_pattern)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Dropout(dropout)(model)
    model = MaxPooling2D(pool_size=(nb_pool, nb_pool))(model)
    model = BatchNormalization()(model)
    # model = Activation('relu')(model)

    for i in range(1, nb_layer):
        model = Conv2D(nb_filters, nb_conv, nb_conv, name='conv2d_' + str(i))(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = Dropout(dropout)(model)
        model = MaxPooling2D(pool_size=(nb_pool, nb_pool))(model)
        model = BatchNormalization()(model)
    # model = Activation('relu')(model)

    model = Dense(hidden, activation='relu')(model)
    model = Activation('relu')(model)
    # model.add(Activation('sigmoid'))
    model = Dropout(dropout)(model)
    out = Dense(nb_classes, name='dense_output', activation='linear')(model)

    model = Model(inputs=[input_pattern], outputs=out)
    model.summary()

    model.compile(loss='mean_squared_error', optimizer=opt)
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                              verbose=0, validation_data=[X_val, Y_val], shuffle=True, metric="mse")

    return history, model


def distortion_model(X_train, Y_train, X_val, Y_val, params):
    print("In", len(X_train))
    # nb_conv = 4
    # nb_pool = 2
    batch_size = 64
    nb_epoch = 1000
    opt = keras.optimizers.Adam()

    nb_filters = params['nb_filters']
    nb_conv = params['nb_conv']
    nb_pool = params['nb_pool']
    nb_layer = params['nb_layer']
    dropout = params['dropout']
    hidden = params['nb_hidden']

    nb_classes = Y_train.shape[1]

    model = Sequential()
    input_shape = (8, 8, 1)

    # build the rest of the network
    model.add(InputLayer(input_shape=input_shape))
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                     padding='valid',
                     input_shape=input_shape, name='conv2d_0'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Activation('sigmoid'))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    for i in range(1, nb_layer):
        model.add(Conv2D(nb_filters, nb_conv, nb_conv, name='conv2d_' + str(i)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # model.add(Activation('sigmoid'))
        model.add(Dropout(dropout))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
    model.add(Flatten())

    model.add(Dense(hidden))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Activation('sigmoid'))
    model.add(Dropout(dropout))
    model.add(Dense(nb_classes, name='dense_output'))
    model.add(Activation('linear'))
    # model.add(Activation('sigmoid'))
    model.summary()

    model.compile(loss='mean_squared_error', optimizer=opt)
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                              verbose=0, validation_data=[X_val, Y_val], shuffle=True)

    return history, model


def model_cnn_masked(x_train, y_train, params):

    # build the rest of the network
    nb_filters = params['nb_filters']
    nb_conv = params['nb_conv']
    nb_pool = params['nb_pool']
    reg_val = params['reg_val']
    nb_layer = params['nb_layer']
    hidden = params['hidden']
    dropout = params['dropout']

    nb_classes = y_train.shape[1]
    (nb_rows, nb_cols, xx) = x_train[0][0].shape

    nb_features = x_train[1].shape[1]
    input_shape = (nb_rows, nb_cols, 2)

    input_pattern = Input(shape=input_shape, name='input1')
    input_geo = Input(shape=(nb_features,), name='input2')

    model = Conv2D(nb_filters, (nb_conv, nb_conv),
                   padding='valid',
                   input_shape=input_shape, kernel_regularizer=regularizers.l2(reg_val),
                   name='conv2d_0')(input_pattern)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    # model.add(Activation('sigmoid'))
    model = Dropout(dropout)(model)
    model = MaxPooling2D(pool_size=(nb_pool, nb_pool))(model)
    model = BatchNormalization()(model)

    for i in range(1, nb_layer):
        model = Conv2D(nb_filters, nb_conv, nb_conv, kernel_regularizer=regularizers.l2(reg_val), name='conv2d_' + str(i))(
            model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = Dropout(dropout)(model)
        model = MaxPooling2D(pool_size=(nb_pool, nb_pool))(model)
        model = BatchNormalization()(model)

    model = GlobalMaxPool2D()(model)
    model = concatenate([model, input_geo])

    model = Dense(hidden, activation='relu', kernel_regularizer=regularizers.l2(reg_val))(model)
    model = Activation('relu')(model)
    model = Dropout(dropout)(model)
    out = Dense(nb_classes, name='dense_output', activation='linear')(model)

    print("Creating hydrid model")
    model_final = Model(inputs=[input_pattern, input_geo], outputs=out)
    model_final.summary()

    G = len([d for d in device_lib.list_local_devices() if d.device_type == 'GPU'])
    # check to see if we are compiling using just a single GPU
    if G <= 1:
        print("[INFO] training with 1 GPU...")
        model_out = model_final

    # otherwise, we are compiling using multiple GPUs
    elif G > 1:
        print("[INFO] training with {} GPUs...".format(G))
        # we'll store a copy of the model on *every* GPU and then combine
        # the results from the gradient updates on the CPU
        with tf.device("/cpu:0"):
            # initialize the model
            model_final_ini = model_final
        # make the model parallel
        model_out = multi_gpu_model(model_final_ini, gpus=G)

    return model_out
