import binvox_rw
import os
import pydicom

# contains [x,y,z,size]
aneurysm_coordinates = dict()
aneurysm_coordinates = {'Patient_C07_MRA': [[212, 170, 66, 8.1]],
                        'Patient_05_MRA': [[222, 44, 271, 3.2]],
                        'Patient_14_MRA': [[178, 152, 78, 25.4]],
                        'Patient_15_MRA': [[218, 65, 265, 16.2]],
                        'Patient_16_MRA': [
    [176, 42, 273, 6.4], [272, 45, 259, 5.5]]}


def import_dicoms():
    dicoms = []

    for file in os.listdir(os.getcwd() + '/data'):
        if ".dcm" in file:
            ds = pydicom.dcmread(os.getcwd() + '/data/' + file)
            # to find the patient in the dictionary
            file = file.replace('.dcm', '')
            # append the aneurysm coordinates to the dicom
            ds.aneurysm = aneurysm_coordinates[file]
            dicoms.append(ds)

    print('Import done.')
    return dicoms
