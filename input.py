import binvox_rw
import os

'''
This function imports the voxelized data
@todo: split into train and test data...
'''


class Object(object):
    pass


def import_voxel():
    images = []
    labels = []
    # load all voxelized files
    for file in os.listdir(os.getcwd() + '/data/C_Patient_Data_Voxelized'):
        with open(os.getcwd() + '/data/C_Patient_Data_Voxelized/' + file, 'rb') as voxel:
            images.append(binvox_rw.read_as_3d_array(voxel).data)
            labels.append(True)

    # put the train data into an object
    train = Object()
    train.images = images
    train.labels = labels

    # create the data object
    data = Object()
    data.train = train
    data.test = []
    return data
