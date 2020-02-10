import numpy as np
import glob
import re
import os
from scipy.spatial import distance
from scipy.interpolate import griddata
from weldAI.utils import coord_nodes
#import matplotlib.pyplot as plt


def pattern_sequence(pattern_folder):

    features_matrix = np.zeros(64)
    features_matrix_dist = np.zeros(64)

    pattern_names = []

    [ini_coord_x, ini_coord_y, ini_coord_z] = coord_nodes(pattern_folder=None, file_name="Initial-Bottom.rpt")

    xi, yi = np.meshgrid(np.unique(ini_coord_x), np.unique(ini_coord_z))
    centroids_matrix = np.mgrid[(max(ini_coord_x) - min(ini_coord_x)) / 16:max(ini_coord_x) - (max(ini_coord_x) - min(ini_coord_x)) / 16:8j,
                       (max(ini_coord_x) - min(ini_coord_x)) / 16:max(ini_coord_x) - (max(ini_coord_x) - min(ini_coord_x)) / 16:8j]

    centroids_matrix = centroids_matrix.reshape(2, centroids_matrix.shape[1]*centroids_matrix.shape[2]).T
    coord_nodes_chunck = np.array([ini_coord_x, ini_coord_y, ini_coord_z]).T
    distances_matrix = distance.cdist(centroids_matrix, coord_nodes_chunck.mean(axis=0)[[0, 2]].reshape(1, 2), "euclidean")
    distances_matrix_corner = distance.cdist(centroids_matrix, np.array([[min(ini_coord_x), min(ini_coord_x)]]),
                                      "euclidean")
    k = 0
    for folder in glob.glob(pattern_folder + "/Mask_0/Sce*"):
        print(folder)
        pattern_names.append(folder.split(os.sep)[-1])
        files = glob.glob(folder + "/*CH*rpt")
        if len(files) > 0:
            features = np.zeros(64)
            for file in files:
                CH = file.split(os.sep)[-1].split("-")[-1].split(".")[0][2:]
                d = file.split(os.sep)[-1].split("-")[0].split(".")[0][1:]
                features[int(CH)-1] = int(d)
            print("Using distortion at scenario", glob.glob(folder + "/D64*"))
            distortions_data = open(glob.glob(folder + "/D64*rpt")[0], "r").read()

            distortions_coord_x = np.array(distortions_data[
                                                  re.search("-------------------------------------------------------------------------------------------------",
                                                            distortions_data).end(): re.search("Minimum", distortions_data).
                                                  start()].split()[3:][::6]).astype(float)

            distortions_coord_y = np.array(distortions_data[
                                                  re.search("-------------------------------------------------------------------------------------------------",
                                                            distortions_data).end(): re.search("Minimum", distortions_data).
                                                  start()].split()[4:][::6]).astype(float)

            distortions_coord_z = np.array(distortions_data[
                                                  re.search("-------------------------------------------------------------------------------------------------",
                                                            distortions_data).end(): re.search("Minimum", distortions_data).
                                                  start()].split()[5:][::6]).astype(float)

            zi = griddata((ini_coord_x + distortions_coord_x, ini_coord_z + distortions_coord_z),
                          ini_coord_y + distortions_coord_y,
                          (xi, yi), method='cubic')

            distortions_coord_y = zi.flatten()
            distortion_chuck_x = xi.flatten().reshape(xi.flatten().shape[0], 1)
            distortion_chuck_y = distortions_coord_y.reshape(distortions_coord_y.shape[0], 1)
            distortion_chuck_z = yi.flatten().reshape(yi.flatten().shape[0], 1)

            if not k:
                features_matrix = features
                features_matrix_dist = distances_matrix[features.astype(int) - 1].T
                features_matrix_dist_corner = distances_matrix_corner[features.astype(int) - 1].T

                distortion_chuck_matrix_x = distortion_chuck_x
                distortion_chuck_matrix_y = distortion_chuck_y
                distortion_chuck_matrix_z = distortion_chuck_z
                k = 1
            else:
                features_matrix = np.vstack((features_matrix, features))
                features_matrix_dist = np.vstack((features_matrix_dist, distances_matrix[features.astype(int) - 1].T))
                features_matrix_dist_corner = np.vstack((features_matrix_dist_corner, distances_matrix_corner[features.astype(int) - 1].T))

                distortion_chuck_matrix_x = np.hstack((distortion_chuck_matrix_x, distortion_chuck_x))
                distortion_chuck_matrix_y = np.hstack((distortion_chuck_matrix_y, distortion_chuck_y))
                distortion_chuck_matrix_z = np.hstack((distortion_chuck_matrix_z, distortion_chuck_z))

        else:
            print("No data found in folder", folder)
    print(features_matrix.shape, features_matrix_dist.shape, features_matrix_dist_corner.shape)
    features_matrix = np.hstack((features_matrix, features_matrix_dist, features_matrix_dist_corner))

    return features_matrix, [distortion_chuck_matrix_x, distortion_chuck_matrix_y, distortion_chuck_matrix_z], \
           [ini_coord_x, ini_coord_y, ini_coord_z], np.array(pattern_names)


def pattern_sequence_masked(pattern_folder=None):

    features_matrix = np.zeros(64)
    features_matrix_dist = np.zeros(64)
    pattern_names = []

    file = pattern_folder + "Initial-Bottom.rpt"

    [ini_coord_x, ini_coord_y, ini_coord_z] = coord_nodes(
        pattern_folder=None, file_name="Initial-Bottom.rpt")

    xi, yi = np.meshgrid(np.unique(ini_coord_x), np.unique(ini_coord_z))

    centroids_matrix = np.mgrid[(max(ini_coord_x) - min(ini_coord_x)) / 16:max(ini_coord_x) - (
                max(ini_coord_x) - min(ini_coord_x)) / 16:8j,
                       (max(ini_coord_x) - min(ini_coord_x)) / 16:max(ini_coord_x) - (
                                   max(ini_coord_x) - min(ini_coord_x)) / 16:8j]

    centroids_matrix = centroids_matrix.reshape(2, centroids_matrix.shape[1] * centroids_matrix.shape[2]).T
    coord_nodes_chunck = np.array([ini_coord_x, ini_coord_y, ini_coord_z]).T
    distances_matrix = distance.cdist(centroids_matrix, coord_nodes_chunck.mean(axis=0)[[0, 2]].reshape(1, 2),
                                      "euclidean")
    distances_matrix_corner = distance.cdist(centroids_matrix, np.array([[min(ini_coord_x), min(ini_coord_x)]]),
                                             "euclidean")
    ll = 0
    for folder_mask in glob.glob(pattern_folder + "/Mask*"):
        mask = np.loadtxt(folder_mask + "/mask.tab", delimiter=",")

        k = 0
        for folder in glob.glob(folder_mask + "/Scenario*"):
            print(folder)
            pattern_names.append(folder_mask.split(os.sep)[-1] + "_" + folder.split(os.sep)[-1])
            files = glob.glob(folder + "/*CH*rpt")
            if len(files) > 0:
                features = np.zeros(64)
                for file in files:
                    # print(file)
                    CH = file.split(os.sep)[-1].split("-")[-1].split(".")[0][2:]
                    d = file.split(os.sep)[-1].split("-")[0].split(".")[0][1:]
                    # print(CH, d)
                    features[int(CH) - 1] = int(d)
                print("Using distortion at scenario", glob.glob(folder + "/D64*"))
                distortions_data = open(glob.glob(folder + "/D64*rpt")[0], "r").read()

                distortions_coord_x = np.array(distortions_data[
                                               re.search(
                                                   "-------------------------------------------------------------------------------------------------",
                                                   distortions_data).end(): re.search("Minimum", distortions_data).
                                               start()].split()[3:][::6]).astype(float)

                distortions_coord_y = np.array(distortions_data[
                                               re.search(
                                                   "-------------------------------------------------------------------------------------------------",
                                                   distortions_data).end(): re.search("Minimum", distortions_data).
                                               start()].split()[4:][::6]).astype(float)

                distortions_coord_z = np.array(distortions_data[
                                               re.search(
                                                   "-------------------------------------------------------------------------------------------------",
                                                   distortions_data).end(): re.search("Minimum", distortions_data).
                                               start()].split()[5:][::6]).astype(float)

                zi = griddata((ini_coord_x + distortions_coord_x, ini_coord_z + distortions_coord_z),
                              ini_coord_y + distortions_coord_y,
                              (xi, yi), method='cubic')

                distortions_coord_y = zi.flatten()
                distortion_chuck_x = xi.flatten().reshape(xi.flatten().shape[0], 1)
                distortion_chuck_y = distortions_coord_y.reshape(distortions_coord_y.shape[0], 1)
                distortion_chuck_z = yi.flatten().reshape(yi.flatten().shape[0], 1)

                if not k:
                    features_matrix = np.hstack(
                        (features.reshape(1, len(features)), mask.reshape(1, mask.shape[0] * mask.shape[1])))
                    features_matrix_dist = distances_matrix[features.astype(int) - 1].T
                    features_matrix_dist_corner = distances_matrix_corner[features.astype(int) - 1].T

                    distortion_chuck_matrix_x = distortion_chuck_x
                    distortion_chuck_matrix_y = distortion_chuck_y
                    distortion_chuck_matrix_z = distortion_chuck_z
                    k = 1
                else:
                    features_matrix = np.vstack((features_matrix, np.hstack(
                        (features.reshape(1, len(features)), mask.reshape(1, mask.shape[0] * mask.shape[1])))))
                    features_matrix_dist = np.vstack(
                        (features_matrix_dist, distances_matrix[features.astype(int) - 1].T))
                    features_matrix_dist_corner = np.vstack(
                        (features_matrix_dist_corner, distances_matrix_corner[features.astype(int) - 1].T))

                    distortion_chuck_matrix_x = np.hstack((distortion_chuck_matrix_x, distortion_chuck_x))
                    distortion_chuck_matrix_y = np.hstack((distortion_chuck_matrix_y, distortion_chuck_y))
                    distortion_chuck_matrix_z = np.hstack((distortion_chuck_matrix_z, distortion_chuck_z))

            else:
                print("No data found in folder", folder)

        if not ll:
            features_matrix_full = np.hstack((features_matrix, features_matrix_dist, features_matrix_dist_corner))

            distortion_chuck_matrix_x_full = distortion_chuck_matrix_x
            distortion_chuck_matrix_y_full = distortion_chuck_matrix_y
            distortion_chuck_matrix_z_full = distortion_chuck_matrix_z

            ll = 1
        else:
            features_matrix_full = np.vstack(
                (features_matrix_full, np.hstack((features_matrix, features_matrix_dist, features_matrix_dist_corner))))

            distortion_chuck_matrix_x_full = np.hstack((distortion_chuck_matrix_x_full, distortion_chuck_matrix_x))
            distortion_chuck_matrix_y_full = np.hstack((distortion_chuck_matrix_y_full, distortion_chuck_matrix_y))
            distortion_chuck_matrix_z_full = np.hstack((distortion_chuck_matrix_z_full, distortion_chuck_matrix_z))

        print(features_matrix_full.shape)

    # print(features_matrix.shape)
    print(features_matrix_full.shape)
    return features_matrix_full, [distortion_chuck_matrix_x_full, distortion_chuck_matrix_y_full,
                                  distortion_chuck_matrix_z_full], \
           [ini_coord_x, ini_coord_y, ini_coord_z], np.array(pattern_names)


def distortion_to_image(pattern_folder=None):

    [ini_coord_x, ini_coord_y, ini_coord_z] = coord_nodes(
        pattern_folder=pattern_folder, file_name="Initial-Bottom.rpt")

    xi, yi = np.meshgrid(np.unique(ini_coord_x), np.unique(ini_coord_z))

    data = read_distortion(pattern_folder=pattern_folder)
    for filename in data.keys():
        zi = griddata((data[filename][0], data[filename][1]),
                      data[filename][2],
                      (xi, yi), method='cubic')
        plt.imshow(zi, origin='lower', interpolation='none')
        plt.savefig('images/' + filename + '_img.png')


def read_distortion(pattern_folder=None):

    [ini_coord_x, ini_coord_y, ini_coord_z] = coord_nodes(
        pattern_folder="C:/Users/mllamosa/Dropbox/2018/Applus/welding patterns all/", file_name="Initial-Bottom.rpt")

    xyz = {}
    for folder_mask in glob.glob(pattern_folder + "/Mask*"):
        mask = np.loadtxt(folder_mask + "/mask.tab", delimiter=",")
        print(mask.shape)
        for folder in glob.glob(folder_mask + "/Scenario*"):
            print(folder)
            files = glob.glob(folder + "/*CH*rpt")
            if len(files) > 0:
                features = np.zeros(64)
                for file in files:
                    # print(file)
                    CH = file.split(os.sep)[-1].split("-")[-1].split(".")[0][2:]
                    d = file.split(os.sep)[-1].split("-")[0].split(".")[0][1:]
                    # print(CH, d)
                    features[int(CH) - 1] = int(d)
                print("Using distortion at scenario", glob.glob(folder + "/D64*"))
                distortions_data = open(glob.glob(folder + "/D64*rpt")[0], "r").read()

                distortions_coord_x = np.array(distortions_data[
                                               re.search(
                                                   "-------------------------------------------------------------------------------------------------",
                                                   distortions_data).end(): re.search("Minimum", distortions_data).
                                               start()].split()[3:][::6]).astype(float)

                distortions_coord_y = np.array(distortions_data[
                                               re.search(
                                                   "-------------------------------------------------------------------------------------------------",
                                                   distortions_data).end(): re.search("Minimum", distortions_data).
                                               start()].split()[4:][::6]).astype(float)

                distortions_coord_z = np.array(distortions_data[
                                               re.search(
                                                   "-------------------------------------------------------------------------------------------------",
                                                   distortions_data).end(): re.search("Minimum", distortions_data).
                                               start()].split()[5:][::6]).astype(float)

                xyz[folder_mask.split(os.sep)[-1] + "_" + folder.split(os.sep)[-1]] = np.array([ini_coord_x + distortions_coord_x, ini_coord_z + distortions_coord_z, ini_coord_y + distortions_coord_y])
    return xyz


def image_plot(folder):

        [ini_coord_x, ini_coord_y, ini_coord_z] = coord_nodes(
            pattern_folder="C:/Users/mllamosa/Dropbox/2018/Applus/welding patterns all/",
            file_name="Initial-Bottom.rpt")

        xi, yi = np.meshgrid(np.unique(ini_coord_x), np.unique(ini_coord_z))

        name = folder.split(os.sep)[-2] + "_" + folder.split(os.sep)[-1]
        files = glob.glob(folder + "/*CH*rpt")
        if len(files) > 0:
            features = np.zeros(64)
            for file in files:
                # print(file)
                CH = file.split(os.sep)[-1].split("-")[-1].split(".")[0][2:]
                d = file.split(os.sep)[-1].split("-")[0].split(".")[0][1:]
                # print(CH, d)
                features[int(CH) - 1] = int(d)
            print("Using distortion at scenario", glob.glob(folder + "/D64*rpt"))

            try:
                distortions_data = open(glob.glob(folder + "/D64*rpt")[0], "r").read()

            except IndexError:
                print(glob.glob(folder + "/D64*rpt"))

            distortions_coord_x = np.array(distortions_data[
                                           re.search(
                                               "-------------------------------------------------------------------------------------------------",
                                               distortions_data).end(): re.search("Minimum", distortions_data).
                                           start()].split()[3:][::6]).astype(float)

            distortions_coord_y = np.array(distortions_data[
                                           re.search(
                                               "-------------------------------------------------------------------------------------------------",
                                               distortions_data).end(): re.search("Minimum", distortions_data).
                                           start()].split()[4:][::6]).astype(float)

            distortions_coord_z = np.array(distortions_data[
                                           re.search(
                                               "-------------------------------------------------------------------------------------------------",
                                               distortions_data).end(): re.search("Minimum", distortions_data).
                                           start()].split()[5:][::6]).astype(float)

            zi = griddata((ini_coord_x + distortions_coord_x, ini_coord_z + distortions_coord_z),
                          ini_coord_y + distortions_coord_y,
                          (xi, yi), method='cubic')

            plt.imshow(zi, origin='lower', interpolation='none')
            plt.savefig('images/' + name + '_img.png')

        return name


def distortion_to_image_parallel(pattern_folder="C:/Users/mllamosa/Dropbox/2018/Applus/welding patterns all/"):

    import multiprocessing as mp

    folder_parent = pattern_folder
    for folder_mask in glob.glob(folder_parent + "/Mask*"):
        folders = glob.glob(folder_mask + "/Scenario*")

        pool = mp.Pool(4)  # pass the amount of processes you want
        results = pool.map(image_plot, folders)

        # pool takes a worker function and input data
        # you usually need to wait for all the subprocesses done their work before
        #using the data; so you don't work on partial data.
        pool.close()
        pool.join()


def pattern_sequence_masked_parallel_(pattern_folder="C:/Users/mllamosa/Dropbox/2018/Applus/welding patterns all/"):
    folder_parent = pattern_folder
    features_matrix = np.zeros(64)
    features_matrix_dist = np.zeros(64)
    pattern_names = []

    file = pattern_folder + "Initial-Bottom.rpt"

    [ini_coord_x, ini_coord_y, ini_coord_z] = coord_nodes(
        pattern_folder="C:/Users/mllamosa/Dropbox/2018/Applus/welding patterns all/", file_name="Initial-Bottom.rpt")

    xi, yi = np.meshgrid(np.unique(ini_coord_x), np.unique(ini_coord_z))

    centroids_matrix = np.mgrid[(max(ini_coord_x) - min(ini_coord_x)) / 16:max(ini_coord_x) - (
                max(ini_coord_x) - min(ini_coord_x)) / 16:8j,
                       (max(ini_coord_x) - min(ini_coord_x)) / 16:max(ini_coord_x) - (
                                   max(ini_coord_x) - min(ini_coord_x)) / 16:8j]

    centroids_matrix = centroids_matrix.reshape(2, centroids_matrix.shape[1] * centroids_matrix.shape[2]).T
    coord_nodes_chunck = np.array([ini_coord_x, ini_coord_y, ini_coord_z]).T
    distances_matrix = distance.cdist(centroids_matrix, coord_nodes_chunck.mean(axis=0)[[0, 2]].reshape(1, 2),
                                      "euclidean")
    distances_matrix_corner = distance.cdist(centroids_matrix, np.array([[min(ini_coord_x), min(ini_coord_x)]]),
                                             "euclidean")
    ll = 0
    for folder_mask in glob.glob(folder_parent + "/Mask*"):
        mask = np.loadtxt(folder_mask + "/mask.tab", delimiter=",")

        k = 0
        for folder in glob.glob(folder_mask + "/Scenario*"):
            print(folder)
            pattern_names.append(folder_mask.split(os.sep)[-1] + "_" + folder.split(os.sep)[-1])
            files = glob.glob(folder + "/*CH*rpt")
            if len(files) > 0:
                features = np.zeros(64)
                for file in files:
                    # print(file)
                    CH = file.split(os.sep)[-1].split("-")[-1].split(".")[0][2:]
                    d = file.split(os.sep)[-1].split("-")[0].split(".")[0][1:]
                    # print(CH, d)
                    features[int(CH) - 1] = int(d)
                print("Using distortion at scenario", glob.glob(folder + "/D64*"))
                distortions_data = open(glob.glob(folder + "/D64*rpt")[0], "r").read()

                distortions_coord_x = np.array(distortions_data[
                                               re.search(
                                                   "-------------------------------------------------------------------------------------------------",
                                                   distortions_data).end(): re.search("Minimum", distortions_data).
                                               start()].split()[3:][::6]).astype(float)

                distortions_coord_y = np.array(distortions_data[
                                               re.search(
                                                   "-------------------------------------------------------------------------------------------------",
                                                   distortions_data).end(): re.search("Minimum", distortions_data).
                                               start()].split()[4:][::6]).astype(float)

                distortions_coord_z = np.array(distortions_data[
                                               re.search(
                                                   "-------------------------------------------------------------------------------------------------",
                                                   distortions_data).end(): re.search("Minimum", distortions_data).
                                               start()].split()[5:][::6]).astype(float)

                zi = griddata((ini_coord_x + distortions_coord_x, ini_coord_z + distortions_coord_z),
                              ini_coord_y + distortions_coord_y,
                              (xi, yi), method='cubic')

                distortions_coord_y = zi.flatten()
                distortion_chuck_x = xi.flatten().reshape(xi.flatten().shape[0], 1)
                distortion_chuck_y = distortions_coord_y.reshape(distortions_coord_y.shape[0], 1)
                distortion_chuck_z = yi.flatten().reshape(yi.flatten().shape[0], 1)

                if not k:
                    # print(features.reshape(len(features), 1).shape, mask.reshape(1, mask.shape[0]*mask.shape[1]).shape)
                    features_matrix = np.hstack(
                        (features.reshape(1, len(features)), mask.reshape(1, mask.shape[0] * mask.shape[1])))
                    features_matrix_dist = distances_matrix[features.astype(int) - 1].T
                    features_matrix_dist_corner = distances_matrix_corner[features.astype(int) - 1].T

                    distortion_chuck_matrix_x = distortion_chuck_x
                    distortion_chuck_matrix_y = distortion_chuck_y
                    distortion_chuck_matrix_z = distortion_chuck_z
                    k = 1
                else:
                    # print(features_matrix.shape, features.shape, mask.shape)
                    features_matrix = np.vstack((features_matrix, np.hstack(
                        (features.reshape(1, len(features)), mask.reshape(1, mask.shape[0] * mask.shape[1])))))
                    features_matrix_dist = np.vstack(
                        (features_matrix_dist, distances_matrix[features.astype(int) - 1].T))
                    features_matrix_dist_corner = np.vstack(
                        (features_matrix_dist_corner, distances_matrix_corner[features.astype(int) - 1].T))

                    distortion_chuck_matrix_x = np.hstack((distortion_chuck_matrix_x, distortion_chuck_x))
                    distortion_chuck_matrix_y = np.hstack((distortion_chuck_matrix_y, distortion_chuck_y))
                    distortion_chuck_matrix_z = np.hstack((distortion_chuck_matrix_z, distortion_chuck_z))

            else:
                print("No data found in folder", folder)

        if not ll:
            features_matrix_full = np.hstack((features_matrix, features_matrix_dist, features_matrix_dist_corner))

            distortion_chuck_matrix_x_full = distortion_chuck_matrix_x
            distortion_chuck_matrix_y_full = distortion_chuck_matrix_y
            distortion_chuck_matrix_z_full = distortion_chuck_matrix_z

            ll = 1
        else:
            features_matrix_full = np.vstack(
                (features_matrix_full, np.hstack((features_matrix, features_matrix_dist, features_matrix_dist_corner))))

            distortion_chuck_matrix_x_full = np.hstack((distortion_chuck_matrix_x_full, distortion_chuck_matrix_x))
            distortion_chuck_matrix_y_full = np.hstack((distortion_chuck_matrix_y_full, distortion_chuck_matrix_y))
            distortion_chuck_matrix_z_full = np.hstack((distortion_chuck_matrix_z_full, distortion_chuck_matrix_z))

        print(features_matrix_full.shape)

    # print(features_matrix.shape)
    print(features_matrix_full.shape)
    return features_matrix_full, [distortion_chuck_matrix_x_full, distortion_chuck_matrix_y_full,
                                  distortion_chuck_matrix_z_full], \
           [ini_coord_x, ini_coord_y, ini_coord_z], np.array(pattern_names)


def pattern_sequence_masked_single(scenario_folder):

        [ini_coord_x, ini_coord_y, ini_coord_z] = coord_nodes(
            pattern_folder="C:/Users/mllamosa/Dropbox/2018/Applus/welding patterns all/", file_name="Initial-Bottom.rpt")

        xi, yi = np.meshgrid(np.unique(ini_coord_x), np.unique(ini_coord_z))

        centroids_matrix = np.mgrid[(max(ini_coord_x) - min(ini_coord_x)) / 16:max(ini_coord_x) - (
                    max(ini_coord_x) - min(ini_coord_x)) / 16:8j,
                           (max(ini_coord_x) - min(ini_coord_x)) / 16:max(ini_coord_x) - (
                                       max(ini_coord_x) - min(ini_coord_x)) / 16:8j]

        centroids_matrix = centroids_matrix.reshape(2, centroids_matrix.shape[1] * centroids_matrix.shape[2]).T
        coord_nodes_chunck = np.array([ini_coord_x, ini_coord_y, ini_coord_z]).T
        distances_matrix = distance.cdist(centroids_matrix, coord_nodes_chunck.mean(axis=0)[[0, 2]].reshape(1, 2),
                                          "euclidean")
        distances_matrix_corner = distance.cdist(centroids_matrix, np.array([[min(ini_coord_x), min(ini_coord_x)]]),
                                                 "euclidean")

        mask = np.loadtxt(scenario_folder + "/../mask.tab", delimiter=",")

        name = scenario_folder.split(os.sep)[-2] + "_" + scenario_folder.split(os.sep)[-1]

        files = glob.glob(scenario_folder + "/*CH*rpt")

        features = np.zeros(64)
        for file in files:
            # print(file)
            CH = file.split(os.sep)[-1].split("-")[-1].split(".")[0][2:]
            d = file.split(os.sep)[-1].split("-")[0].split(".")[0][1:]
            # print(CH, d)
            features[int(CH) - 1] = int(d)

        print("Using distortion at scenario", glob.glob(scenario_folder))
        print("Using distortion at scenario", glob.glob(scenario_folder + "/D64*rpt")[-1])
        distortions_data = open(glob.glob(scenario_folder + "/D64*rpt")[0], "r").read()

        distortions_coord_x = np.array(distortions_data[
                                       re.search(
                                           "-------------------------------------------------------------------------------------------------",
                                           distortions_data).end(): re.search("Minimum", distortions_data).
                                       start()].split()[3:][::6]).astype(float)

        distortions_coord_y = np.array(distortions_data[
                                       re.search(
                                           "-------------------------------------------------------------------------------------------------",
                                           distortions_data).end(): re.search("Minimum", distortions_data).
                                       start()].split()[4:][::6]).astype(float)

        distortions_coord_z = np.array(distortions_data[
                                       re.search(
                                           "-------------------------------------------------------------------------------------------------",
                                           distortions_data).end(): re.search("Minimum", distortions_data).
                                       start()].split()[5:][::6]).astype(float)

        zi = griddata((ini_coord_x + distortions_coord_x, ini_coord_z + distortions_coord_z),
                      ini_coord_y + distortions_coord_y,
                      (xi, yi), method='cubic')

        distortions_coord_y = zi.flatten()
        distortion_chuck_x = xi.flatten().reshape(xi.flatten().shape[0], 1)
        distortion_chuck_y = distortions_coord_y.reshape(distortions_coord_y.shape[0], 1)
        distortion_chuck_z = yi.flatten().reshape(yi.flatten().shape[0], 1)

        features_matrix = np.hstack(
            (features.reshape(1, len(features)), mask.reshape(1, mask.shape[0] * mask.shape[1])))
        features_matrix_dist = distances_matrix[features.astype(int) - 1].T
        features_matrix_dist_corner = distances_matrix_corner[features.astype(int) - 1].T

        return [np.hstack((features_matrix, features_matrix_dist, features_matrix_dist_corner)),
                (distortion_chuck_x, distortion_chuck_y, distortion_chuck_z), [ini_coord_x, ini_coord_y, ini_coord_z], name]


def pattern_sequence_masked_parallel(pattern_folder="C:/Users/mllamosa/Dropbox/2018/Applus/welding patterns all/"):

    import multiprocessing as mp

    folders = glob.glob(pattern_folder + "//Mask*//Scenario*")
    pool = mp.Pool(4)  # pass the amount of processes you want
    features_list = pool.map(pattern_sequence_masked_single, folders)

    pool.close()
    pool.join()
    return features_list

