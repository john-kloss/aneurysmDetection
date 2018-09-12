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


def plot_full_dicom(pixel_array, predicted_aneurysm):
    dims = pixel_array.shape
    pixel_array = np.where(pixel_array > 0.5)
    predicted_aneurysm = np.where(predicted_aneurysm > 0.5)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim([0, dims[0]])
    ax.set_ylim([0, dims[1]])
    ax.set_zlim([0, dims[2]])

    ax.scatter(
        predicted_aneurysm[0],
        predicted_aneurysm[1],
        predicted_aneurysm[2],
        zdir="z",
        c="red",
        alpha=0.9,
        marker=".",
    )

    ax.scatter(
        pixel_array[0],
        pixel_array[1],
        pixel_array[2],
        zdir="z",
        c="b",
        alpha=0.05,
        marker="."
        # edgecolors='none'
    )

    plt.show()


path = os.getcwd() + "/data/processed/test/"


def predict():
    net = unet_model_3d((1, 64, 64, 64))
    net.load_weights("./data/logs/network_weights_loss_new_processed.h5")

    for patient in os.listdir(path):
        if not ".h" in patient:
            continue

        f = h5py.File(path + patient, "r")
        amount_of_subvolumes = len(f["images/images"])

        pixel_array = np.array(f["pixel/pixel"])
        predicted_aneurysm = np.zeros((pixel_array.shape))

        for i in range(amount_of_subvolumes):
            images = np.array(np.reshape(f["images/images"][i], (1, 1, 64, 64, 64)))
            labels = np.array(np.reshape(f["labels/labels"][i], (1, 1, 64, 64, 64)))
            cuts = f["cuts/cuts"][i]

            prediction = net.predict(images, batch_size=1, verbose=1)

            # overlay each predicted aneurysm on top of dicom pixel array
            for x in range(64):
                for y in range(64):
                    for z in range(64):
                        if prediction[0][0][x][y][z] > 0.9:
                            predicted_aneurysm[x + cuts[0] - 32][y + cuts[1] - 32][
                                z + cuts[2] - 32
                            ] = 1

        plot_full_dicom(pixel_array, predicted_aneurysm)

