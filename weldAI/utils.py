
"""Data reading and manipulation"""
from xlrd import open_workbook
import numpy as np
import pandas as pd
from scipy.spatial import distance
import glob
from sklearn.metrics import pairwise_distances


class MinMax:

    def __init__(self, parameter=None):
        self.parameters=None

    def copy(self):
        return minmax()

    def map_data(self, data):
        max = np.amax(data, axis=0)
        min = np.amin(data, axis=0)
        datanormalize = 2*(data-min)/(max-min) - 1
        self.parameters = [max, min]
        return datanormalize

    def remap_data(self,data):

        '''Check if normalizing only label (1-D array) or all data matrix features and labels '''

        if len(data.shape)==1 :
            max=self.parameters[0][-1]
            min=self.parameters[1][-1]
            #print "MaxMin",max,min
        else:
            max=self.parameters[0]
            min=self.parameters[1]
            #print "MaxMin",max,min
            #print data
        databacknormalize=((data+1)*(max-min))/2 + min
        #print databacknormalize
        return  databacknormalize

    def transmap_data(self,data):
        #print np.asmatrix(data).shape
    #print self.parameters
        if len(data.shape)==1 :
            max=self.parameters[0][-1]
            min=self.parameters[1][-1]
        else:
            max=self.parameters[0]
            min=self.parameters[1]

        #print "MaxMin",max,min
        #for idx in range(data.shape[1]):
        #    datanormalize[:,idx]=2*(data[:,idx]-min[idx])/(max[idx]-min[idx]) - 1
        datanormalize=2*(data-min)/(max-min) - 1
        return  datanormalize

    def copy(self):
        return  minmax()

    def get_parameters(self):
        return self.parameters


def coord_nodes(pattern_folder="data/Mask_3/", file_name="Scenario1- All Distortion/Initial-Bottom.rpt"):

    import re

    file = pattern_folder + file_name
    distortions_data = open(file, "r").read()

    coord = distortions_data[re.search("------------------------------------------------------------------------------"
                                       "-------------------", distortions_data).end(): re.search("Minimum", distortions_data).start()]

    ini_coord_x = np.array(coord.split()[1:][::6]).astype(float)
    ini_coord_y = np.array(coord.split()[2:][::6]).astype(float)
    ini_coord_z = np.array(coord.split()[3:][::6]).astype(float)

    return [ini_coord_x, ini_coord_y, ini_coord_z]


def node_trench_centroid_distance(distortionfile, trenchcoordfile):

    data_distortion = pd.read_excel(distortionfile)
    xyz_ini = np.array([data_distortion['X'], data_distortion['Y'], data_distortion['Z']]).T

    data_trench = pd.read_excel(trenchcoordfile)
    xyz_trench = np.array([data_trench['X'], data_trench['Y'], data_trench['Z']]).T

    return distance.cdist(xyz_ini, xyz_trench.mean(axis=0).reshape(1, 3), "euclidean")


def node_trench_all_centroid_distance(distortionfile, trenchcoordfile):

    data_distortion = pd.read_excel(distortionfile)
    xyz_ini = np.array([data_distortion['X'], data_distortion['Y'], data_distortion['Z']]).T

    data_trench = pd.read_excel(trenchcoordfile)
    xyz_trench = np.array([data_trench['X'], data_trench['Y'], data_trench['Z']]).T

    return distance.cdist(xyz_ini, xyz_trench.mean(axis=0).reshape(1, 3), "euclidean")


def node_trench_centroid_weighted_distance(distortionfile, trenchcoordfile):

    data_distortion = pd.read_excel(distortionfile)
    xyz_ini = np.array([data_distortion['X'], data_distortion['Y'], data_distortion['Z']]).T

    data_trench = pd.read_excel(trenchcoordfile)
    for trench in data_trench.keys():
        xyz_trench = np.array([data_trench[trench]['X'], data_trench[trench]['Y'], data_trench[trench]['Z']]).T

    return distance.cdist(xyz_ini, xyz_trench.mean(axis=0).reshape(1, 3), "euclidean")


def node_trench_distance_matrix(distortionfile, trenchcoordfile):

    data_distortion = pd.read_excel(distortionfile)
    xyz_ini = np.array([data_distortion['X'], data_distortion['Y'], data_distortion['Z']]).T

    data_trench = pd.read_excel(trenchcoordfile)
    xyz_trench = np.array([data_trench['X'], data_trench['Y'], data_trench['Z']]).T

    return xyz_ini - xyz_trench.mean(axis=0).reshape(1, 3)


def load_distance_features(plate_coord_file, trench_data_file, distortion_folder):

    plate_coord = pd.read_excel(plate_coord_file)
    distance_matrix = pairwise_distances(plate_coord.values)
    edge_ids = np.where(distance_matrix == np.max(distance_matrix))[0]
    edge_coord = plate_coord.values[edge_ids, :]

    # Distance to trenches centroids. Generate centroids for all trenches first
    data_trench = pd.read_excel(trench_data_file).values

    numberOfTrench = int(data_trench.shape[1]/3)
    trench_centroid = []
    for i in range(numberOfTrench):
        trench_centroid.append(data_trench[i:i+3].mean(axis=1))

    trench_number = 0
    for file in glob.glob(distortion_folder + "/*xlsx"):
        print(file)
        data_distortion = pd.read_excel(file)
        xyz_ini = np.array([data_distortion['X'], data_distortion['Y'], data_distortion['Z']]).T
        distance_to_trench = 0.1/distance.cdist(xyz_ini, trench_centroid[0].reshape(1, 3), "euclidean")
        for i in range(1, len(trench_centroid)):
                if i <= trench_number:
                    distance_to_trench = np.hstack((distance_to_trench, 0.1/distance.cdist(xyz_ini, trench_centroid[i].reshape(1, 3), "euclidean")))
                else:
                    distance_to_trench = np.hstack((distance_to_trench, 1/distance.cdist(xyz_ini, trench_centroid[i].reshape(1, 3), "euclidean")))

        # Distance to edge features
        distance_to_edge = distance.cdist(xyz_ini, np.array([edge_coord[0]]), "euclidean")
        for edge in edge_coord[1:]:
            distance_to_edge = np.hstack((distance_to_edge, distance.cdist(xyz_ini, np.array([edge]),
                                                                           "euclidean")))

        xyz_before = np.array([data_distortion['X.1'], data_distortion['Y.1'], data_distortion['Z.1']]).T
        xyz_after = np.array([data_distortion['X.2'], data_distortion['Y.2'], data_distortion['Z.2']]).T

        distortion_before = []
        distortion_after = []
        position_z = []
        distortion_z_after = []
        for j in range(xyz_ini.shape[0]):
            distortion_before.append(distance.euclidean(xyz_ini[j, :], (xyz_ini[j, :] + xyz_before[j, :])))
            distortion_after.append(distance.euclidean(xyz_ini[j, :], (xyz_ini[j, :] + xyz_after[j, :])))
            distortion_z_after.append(distance.euclidean(xyz_ini[j, 2], (xyz_ini[j, 2] + xyz_after[j, 2])))
            position_z.append((xyz_ini[j, 2] + xyz_after[j, 2]))

        position_z = np.array(position_z)

        distortion_feature = position_z
        feature_data = np.hstack((np.arange(0, distance_to_trench.shape[0]).reshape(distance_to_trench.shape[0], 1), distance_to_trench, distance_to_edge, distortion_feature.reshape(len(distortion_feature), 1)))
        feature_data_qsar = np.vstack((np.arange(0, feature_data.shape[1]).reshape(feature_data.shape[1], 1).T, feature_data))
        np.savetxt(file + "_qsar.csv", feature_data_qsar, delimiter=",", fmt="%4.6f")

        if trench_number == 0:
            feature_data_full = feature_data
        else:
            feature_data_full = np.vstack((feature_data_full, feature_data))

        trench_number += 1

    return feature_data_full


def surface_mask(pattern=None, mask=None):

    ids = np.array([np.where(mask[i, :] == 0)[0] for i in range(mask.shape[0])])
    id_selected = np.array([ids[:, i] for i in range(ids.shape[1]) if sum(ids[:, i+1] - ids[:, i]) == len(ids[:, i])]).T

    id_stiffener = np.array(
        [ids[:, i] for i in range(ids.shape[1]) if sum(ids[:, i + 1] - ids[:, i]) != len(ids[:, i])]).T

    if id_stiffener[0][0]:
            id_selected[:, id_discard[0][0] - 1:]

            id_selected[:, 0:id_discard[0][0]]
    else:
        pass

    return id_selected


def stats_scenario(distortion, predictions, distortion_coord, pattern_names,
                       folder_name="C:/Users/mllamosa/Dropbox/2018/Applus/welding patterns all/plots/", tag="train"):
        with open(folder_name + tag + "_full_report.csv", 'w') as fp:
            for i in range(distortion.shape[0]):
                print(tag, "stats for scenario", pattern_names[i])
                print("Y axis(out of the plane) ")
                train_stats = all_stats(distortion[i:i + 1, :], predictions[i:i + 1, :])
                print([val for val in train_stats])
                fp.write(tag + "_" + pattern_names[i] + "," + ",".join(train_stats.astype(str)) + "\n")

        fp.close()


def load_distortion_data(folder, training_fraction=0.8):

    from data.pattern_features_grid import pattern_sequence_masked
    features_matrix, distortion_coord, ini_coord, pattern_names = pattern_sequence_masked(pattern_folder=folder)

    ids = np.array(range(len(pattern_names)))
    np.random.shuffle(ids)
    train_id = ids[0:int(ids.shape[0] * training_fraction)]
    test_id = ids[int(ids.shape[0] * training_fraction):]
    print("Training with", [pattern for pattern in pattern_names[train_id]])
    print("Test with", [pattern for pattern in pattern_names[test_id]])

    X_train = features_matrix[train_id, :]
    X_test = features_matrix[test_id, :]

    Y_train = distortion_coord[1].T[train_id, :]
    Y_test = distortion_coord[1].T[test_id, :]

    return (X_train, Y_train), (X_test, Y_test.flatten())







