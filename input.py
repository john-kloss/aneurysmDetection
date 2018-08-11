import binvox_rw
import os
import pydicom

# this object contains all voxel aneurysm positions
# in the form of [x,y,z,size] for each aneurysm
aneurysm_coordinates = {
    "Patient_C07_MRA": [[66, 170, 212, 8.1]],
    "Patient_05_MRA": [[271, 44, 222, 3.2]],
    "Patient_14_MRA": [[78, 152, 178, 25.4]],
    "Patient_15_MRA": [[265, 65, 218, 16.2]],
    "Patient_16_MRA": [[273, 42, 176, 6.4], [259, 45, 272, 5.5]],
    "Patient_C68_TOF-MRA": [
        [36, 151, 149, 7],
        [34, 141, 183, 5.7],
        [11, 194, 123, 6.7],
    ],
    "Patient_C121_CE-MRA": [],
    # transformed images
    "Patient_C38_MRA": [[51, 98, 212, 6.4]],
    "Patient_C42_MRA": [[]],
    "Patient_C55_MRA": [[]],
    "Patient_C69_MRA": [[]],
    "Patient_C87_CE-MRA": [[97, 168, 248, 5.4]],
    "Patient_C96_CE-MRA": [[179, 79, 153, 14.5]],
    "Patient_C101_CE-MRA": [[]],
    "Patient_C127_TOF-MRA": [[197, 85, 249, 5.5], [203, 112, 203, 4.5]],
}


def import_dicoms():
    dicoms = []

    for file in os.listdir(os.getcwd() + "/data"):
        if ".dcm" in file:
            ds = pydicom.dcmread(os.getcwd() + "/data/" + file)
            # to find the patient in the dictionary
            file = file.replace(".dcm", "")
            # append the aneurysm coordinates to the dicom
            # ds.aneurysm = aneurysm_coordinates[file]
            # dicoms.append(ds)

    print("Import done.")
    return dicoms
