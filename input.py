import binvox_rw
import os
import pydicom

# this object contains all voxel aneurysm positions
# in the form of [x,y,z,size] for each aneurysm
aneurysm_coordinates = {
    'Patient_C07_MRA': [[212, 170, 66, 8.1]],
    'Patient_05_MRA': [[222, 44, 271, 3.2]],
    'Patient_14_MRA': [[178, 152, 78, 25.4]],
    'Patient_15_MRA': [[218, 65, 265, 16.2]],
    'Patient_16_MRA': [[176, 42, 273, 6.4], [272, 45, 259, 5.5]],

    # transformed images
    'Patient_C38_MRA': [[212, 98, 51, 6.4]],
    'Patient_C42_MRA': [[]],
    'Patient_C55_MRA': [[]],
    'Patient_C69_MRA': [[]],
    'Patient_C87_CE-MRA': [[248, 168, 97, 5.4]],
    'Patient_C96_CE-MRA': [[153, 79, 179, 14.5]],
    'Patient_C101_CE-MRA': [[]],
    'Patient_C127_TOF-MRA': [[249, 85, 197, 5.5], [203, 112, 203, 4.5]]

}


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
