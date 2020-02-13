from weldAI.models import model_cnn_masked
import numpy as np
import random
import pickle
from keras import backend as K


def model_eval(pattern=np.array([]), pattern_stiff=np.array([]), scenario_col=8, scenario_row=8):

            feature_input = pattern
            img_rows = scenario_row
            img_cols = scenario_col

            dropout = 0.6
            nb_filters = 24
            nb_conv = 6
            nb_pool = 3
            nb_layer = 0
            hidden = 1000
            reg_val = 0.001
            params = {'nb_filters': nb_filters, 'nb_conv': nb_conv, 'nb_pool': nb_pool,
                      'reg_val': reg_val, 'nb_layer': nb_layer, 'hidden': hidden, 'dropout': dropout}

            stiffener_mask = np.zeros((1, scenario_row*scenario_col))

            stiffener_mask[:, pattern_stiff] = 1
            stiffener_mask = stiffener_mask.reshape(1, scenario_row, scenario_col, 1)

            geometrical_desc = np.array([309.35921119, 268.82265909, 237.99290438, 220.97086466,
                   220.97086577, 237.99290749, 268.82266368, 309.35921677,
                   268.82265909, 220.97086354, 182.2172413, 159.34435516,
                   159.34435671, 182.21724536, 220.97086912, 268.82266551,
                   237.99290438, 182.2172413, 132.5825159, 98.82117189,
                   98.82117439, 132.58252147, 182.21724807, 237.99291163,
                   220.97086466, 159.34435516, 98.82117189, 44.19416825,
                   44.19417382, 98.82117937, 159.34436289, 220.97087247,
                   220.97086577, 159.34435671, 98.82117439, 44.19417382,
                   44.1941794, 98.82118187, 159.34436444, 220.97087358,
                   237.99290749, 182.21724536, 132.58252147, 98.82117937,
                   98.82118187, 132.58252705, 182.21725212, 237.99291474,
                   268.82266368, 220.97086912, 182.21724807, 159.34436289,
                   159.34436444, 182.21725212, 220.9708747, 268.8226701,
                   309.35921677, 268.82266551, 237.99291163, 220.97087247,
                   220.97087358, 237.99291474, 268.8226701, 309.35922235,
                   44.19417382, 98.82117688, 159.3443598, 220.97086912,
                   282.98078557, 345.16753179, 407.45015033, 469.79051182,
                   98.82117688, 132.58252147, 182.21724671, 237.99290956,
                   296.46353064, 356.30482034, 416.927002, 478.0330794,
                   159.3443598, 182.21724671, 220.97086912, 268.8226646,
                   321.73844191, 377.59518667, 435.26213366, 494.1058844,
                   220.97086912, 237.99290956, 268.8226646, 309.35921677,
                   356.30482034, 407.45015033, 461.40072063, 517.27954241,
                   282.98078557, 296.46353064, 321.73844191, 356.30482034,
                   397.74756442, 444.14595011, 494.1058844, 546.65174014,
                   345.16753179, 356.30482034, 377.59518667, 407.45015033,
                   444.14595011, 486.13591207, 532.16832394, 581.28360118,
                   407.45015033, 416.927002, 435.26213366, 461.40072063,
                   494.1058844, 532.16832394, 574.52425971, 620.29478879,
                   469.79051182, 478.0330794, 494.1058844, 517.27954241,
                   546.65174014, 581.28360118, 620.29478879, 662.91260736])

            geometrical_input = geometrical_desc[np.hstack((feature_input, feature_input + img_rows*img_cols)) - 1]

            norm = pickle.load(open("data/norm.p", "rb"))
            geometrical_input_norm = norm.transmap_data(geometrical_input)

            feature_input = feature_input.reshape(feature_input.shape[0], img_rows, img_cols, 1)
            feature_input_norm = np.concatenate((feature_input, stiffener_mask), axis=3)

            x_test = [feature_input_norm[:, :, :], geometrical_input_norm[:, :]]
            y_test = np.array([random.random() for i in range(5041)]).reshape(1, 5041)

            model = model_cnn_masked(x_test, y_test, params)
            model.load_weights('data/model_weights.h5')

            y_score_test = model.predict(x_test)
            K.clear_session()

            y_mean_std = pickle.load(open("data/y_std.p", "rb"))
            distortion_prediction = y_score_test*y_mean_std['y_std'] - y_mean_std['y_mean']

            return distortion_prediction

