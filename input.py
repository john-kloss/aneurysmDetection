import binvox_rw
import os


class Object(object):
    pass


'''
This function imports the voxelized data
'''


def import_single_voxel(fileName):
    with open(os.getcwd() + '/data/obj/' + fileName, 'rb') as voxel:
        return binvox_rw.read_as_3d_array(voxel).data


def import_voxel():
    # load all voxelized files
    images_true = []
    for file in os.listdir(os.getcwd() + '/data/C_Patient_Data_Voxelized'):
        with open(os.getcwd() + '/data/C_Patient_Data_Voxelized/' + file, 'rb') as voxel:
            images_true.append(binvox_rw.read_as_3d_array(voxel).data)

    images_false = []
    for file in os.listdir(os.getcwd() + '/data/D_Data'):
        with open(os.getcwd() + '/data/D_Data/' + file, 'rb') as voxel:
            images_false.append(binvox_rw.read_as_3d_array(voxel).data)

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    # add 80 % of the data to the training set
    for i in range(int(len(images_true)*8/10)):
        train_images.append(images_true[i])
        train_labels.append(True)

    for i in range(int(len(images_false)*8/10)):
        train_images.append(images_false[i])
        train_labels.append(False)

    # and 20% to to the test set
    for i in range(int(len(images_true)*8/10), len(images_true)):
        test_images.append(images_true[i])
        test_labels.append(True)

    for i in range(int(len(images_false)*8/10), len(images_false)):
        test_images.append(images_false[i])
        test_labels.append(False)

    # put the train data into an object

    train = Object()
    train.images = train_images
    train.labels = train_labels

    test = Object()
    test.images = test_images
    test.labels = test_labels
    # create the data object
    data = Object()
    data.train = train
    data.test = test
    return data
