
import os
import preprocessing.augment
import pydicom

class Dicom:
    def __init__(self, patient, aneurysm, pixel_array):
        self.patient = patient
        self.aneurysm = aneurysm
        self.pixel_array = pixel_array
        self.mask = None
        self.augmentations = None
    


# this object contains all voxel aneurysm positions
# in the form of [x,y,z,size] for each aneurysm
aneurysm_coordinates = {
    "Patient_C07_MRA": [[66, 170, 212, 8.1]],
    "Patient_05_MRA": [[271, 44, 222, 3.2]],
    "Patient_14_MRA": [[78, 152, 178, 25.4]],
    "Patient_15_MRA": [[265, 65, 218, 16.2]],
    "Patient_16_MRA": [[273, 42, 176, 6.4], [259, 45, 272, 5.5]],
    "Patient_C121_CE-MRA": [],
    "Patient_21_TOF-MRA": [64, 143, 159, 18.5],
    "Patient_105_MRA": [211, 93, 239, 24],
    "Patient_C37_MRA": [142, 132, 183, 10.8],
    "Patient_C01_MRA": [96, 185, 228, 23.8],
    "Patient_C02_MRA": [38, 178, 175, 15.5],
    "Patient_C74_CE-MRA": [74, 208, 340, 3.5],
    "Patient_C59_TOF-MRA": [150, 169, 187, 6.9],
    "Patient_C68_TOF-MRA": [
        [36, 151, 149, 7],
        [34, 141, 183, 5.7],
        [11, 194, 123, 6.7],
    ],
    "Patient_C82_TOF-MRA": [[61, 129, 195, 5.3], [88, 147, 281, 3.8]],
    "Patient_C86_TOF-MRA": [[54, 182, 132, 24.7], [56, 167, 184, 5.1]],
    "Patient_C87_TOF-MRA": [56, 154, 258, 5.1],
    "Patient_C92_TOF-MRA": [[64, 176, 152, 5.2], [64, 195, 143, 6.5]],
    "Patient_C101_TOF-MRA": [[103, 157, 118, 5.8], [101, 147, 170, 4.3]],
    "Patient_C101_CE-MRA_new": [[191, 36, 121, 4.3], [191, 45, 78, 5.8]],
    "Patient_C108_CE-MRA": [[255, 80, 226, 5.5], [260, 47, 182, 4.5]],
    "Patient_C108_TOF-MRA": [[114, 183, 176, 4.5], [115, 222, 229, 5.5]],
    "Patient_C113_TOF-MRA": [[72, 165, 135, 2.6], [60, 180, 244, 7.3]],
    "Patient_C113_CE-MRA": [[262, 59, 218, 3.9], [269, 55, 135, 2.6]],
    "Patient_C113_CE-MRA_new": [[182, 58, 192, 3.9], [192, 55, 110, 2.6]],
    "Patient_C120_CE-MRA": [[253, 51, 223, 9.1]],
    "Patient_C120_TOF-MRA": [[124, 143, 188, 9.1]],
    "Patient_C121_CE-MRA_new": [[196, 57, 167, 5.7]],
    "Patient_C121_TOF-MRA": [[75, 152, 198, 5.7]],
    "Patient_C138_CE-MRA": [[260, 51, 218, 5.4]],
    "Patient_C138_TOF-MRA": [[102, 150, 196, 5.4]],
    "Patient_C144_CE-MRA": [[214, 90, 207, 6.8]],
    "Patient_C144_TOF-MRA": [[104, 181, 181, 6.8]],
    "Patient_C64_TOF-MRA": [[91, 166, 233, 4.7]],
    # transformed images
    "Patient_C38_MRA": [[51, 98, 212, 6.4]],
    "Patient_C42_MRA": [[]],
    "Patient_C46_MRA": [[]],
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
            dicom_object = Dicom(file, aneurysm_coordinates[file], ds.pixel_array)
            
            dicoms.append(dicom_object)
            #preprocessing.augment.shear_images(dicoms[0],1)
            #ds.PixelData = dicoms[0].shears["pixel_array"][0]
            #ds.save_as(os.getcwd() + "/data/shear.dcm")
    
    print("Import done.")
    return dicoms
