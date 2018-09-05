import os
import numpy as np
import tables
import h5py
from .model import unet_model_3d
from .generator import DataGenerator
from .metrics import dice_coefficient
from sklearn.metrics import log_loss
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_mask(mask):
    plt.imshow(mask[0])
    plt.show()
    plt.imshow(mask[20])
    plt.show()
    plt.imshow(mask[40])
    plt.show()


def plot(prediction, labels):
    aneurysm_pred = np.where(prediction[0][0] > 0.5)
    fig = plt.figure()
    ax = fig.add_subplot(121, projection="3d")
    ax.set_xlim([0, 64])
    ax.set_ylim([0, 64])
    ax.set_zlim([0, 64])
    ax.scatter(
        aneurysm_pred[0],
        aneurysm_pred[1],
        aneurysm_pred[2],
        zdir="z",
        c="red",
        alpha=0.5,
        marker=".",
    )

    aneurysm_pred = np.where(labels[0][0] > 0.5)
    ax = fig.add_subplot(122, projection="3d")
    ax.set_xlim([0, 64])
    ax.set_ylim([0, 64])
    ax.set_zlim([0, 64])
    ax.scatter(
        aneurysm_pred[0],
        aneurysm_pred[1],
        aneurysm_pred[2],
        zdir="z",
        c="red",
        alpha=0.5,
        marker=".",
    )
    plt.show()


path = os.getcwd() + "/data/processed/test/"


def predict():
    net = unet_model_3d((1, 64, 64, 64))
    net.load_weights("./data/logs/network_weights_loss.h5")
    global_tp = 0
    global_fn = 0
    global_fp = 0

    for patient in os.listdir(path):
        if not ".h" in patient:
            continue

        f = h5py.File(path + patient, "r")
        amount_of_subvolumes = len(f["images/images"])

        tp = 0
        fp = 0
        fn = 0

        for i in range(amount_of_subvolumes):
            images = np.array(np.reshape(f["images/images"][i], (1, 1, 64, 64, 64)))
            labels = np.array(np.reshape(f["labels/labels"][i], (1, 1, 64, 64, 64)))
            # if len(np.nonzero(labels)[1]) == 0:
            # continue
            prediction = net.predict(images, batch_size=1, verbose=1)

            highly_conf_predicted = len(np.where(prediction[0][0] > 0.99)[0])
            # plot(prediction, labels)

            # aneurysm in mask -> dice can be considered as measure
            if len(np.nonzero(labels)[1]) != 0:
                dc = 1 - distance.dice(
                    np.reshape(labels, (-1,)), np.reshape(prediction, (-1,))
                )

                if dc > 0.30:
                    # aneurysm detected correctly
                    tp += 1
                    visualize_mask(prediction[0][0])
                    visualize_mask(labels[0][0])
                else:
                    # aneurysm not detected correctly
                    fn += 1
                    visualize_mask(prediction[0][0])
                    visualize_mask(labels[0][0])

            # no aneurysm in mask but in prediction
            elif highly_conf_predicted > 50:
                # check whether this is predicted aneurysm or random activation (check is across one axis only)
                max_index = np.max((np.where(prediction[0][0] > 0.99)[0]))
                min_index = np.min((np.where(prediction[0][0] > 0.99)[0]))
                if max_index - min_index < np.cbrt(highly_conf_predicted) + 5:
                    fp += 1

        # compute precision and recall per patient
        precision = tp + 0.0001 / (tp + fp + 0.0001)
        recall = tp + 0.0001 / (tp + fn + 0.0001)
        print("precision: " + str(precision) + " recall: " + str(recall))

        global_fn += fn
        global_fp += fp
        global_tp += tp

    precision = global_tp / (global_tp + global_fp)
    recall = global_tp / (global_tp + global_fn)
    print("precision: " + str(precision) + " recall: " + str(recall))

