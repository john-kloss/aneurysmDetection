
import os
import preprocessing.augment
import pydicom
import matplotlib.pyplot as plt



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
    "Patient_21_TOF-MRA": [[64, 143, 159, 18.5]],
    "Patient_105_MRA": [[211, 93, 239, 24]],
    "Patient_C37_MRA": [[142, 132, 183, 10.8]],
    "Patient_C01_MRA": [[96, 185, 228, 23.8]],
    "Patient_C02_MRA": [[38, 178, 175, 15.5]],
    "Patient_C74_CE-MRA": [[74, 208, 340, 3.5]],
    "Patient_C59_TOF-MRA": [[150, 169, 187, 6.9]],
    "Patient_C68_TOF-MRA": [
        [36, 151, 149, 7],
        [34, 141, 183, 5.7],
        [11, 194, 123, 6.7],
    ],
    "Patient_C82_TOF-MRA": [[61, 129, 195, 5.3], [88, 147, 281, 3.8]],
    "Patient_C86_TOF-MRA": [[54, 182, 132, 24.7], [56, 167, 184, 5.1]],
    "Patient_C87_TOF-MRA": [[56, 154, 258, 5.1]],
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
    "Patient_C87_CE-MRA": [[97, 168, 248, 5.4]],
    "Patient_C96_CE-MRA": [[179, 79, 153, 14.5]],
    "Patient_C127_TOF-MRA": [[197, 85, 249, 5.5], [203, 112, 203, 4.5]],

    # DSA
    'Patient_03_DSA_256': [[118, 69, 85, 21.3]],
    'Patient_13_DSA': [[47, 98, 107, 16.1], [60, 76, 112, 3.1] ,[70, 95, 117, 3.0]],
    'Patient_C81_DSA1': [[106, 118, 207, 4.8]],
    'Patient_C81_DSA2': [[146, 167, 199, 3.9]],
    'Patient_06_DSA': [[136, 151, 144, 16.6]],
    'Patient_07_DSA_256': [[84, 113, 153, 7.5]],
    'Patient_10_DSA': [[152, 156, 142, 13.0]],
    'Patient_11_DSA': [[103, 102, 142, 31.7]],
    'Patient_09_DSA': [[108, 136, 139, 5.5]],
    'Patient_100_DSA': [[104, 59, 195, 7.1]],
    'Patient_103_DSA': [[19, 87, 174, 16.6]],
    'Patient_104_DSA': [[53, 110, 110, 6.7]],
    'Patient_107_DSA': [[100, 109, 94, 2.6] , [92, 70, 143, 3.1]],
    'Patient_108_DSA': [[44, 121, 119, 4.4]],
    'Patient_109_DSA': [[94, 65, 142, 9.4]],
    'Patient_111_DSA': [[100, 112, 184, 2.1] ,[109, 48, 159, 3.5]],
    'Patient_C102_DSA': [[85, 148, 177, 20.9]],
    'Patient_C86_DSA': [[74, 135, 139, 24.7]],
    'Patient_C97_DSA': [[108, 131, 218, 6.3]],
    'Patient_113_DSA': [[58, 44, 80, 8.7] , [51, 91, 110, 3.5]],
    'Patient_12_DSA': [[86, 68, 191, 18.6]],
    'Patient_14_DSA': [[94, 96, 136, 25.4]],
    'Patient_15_DSA': [[53, 108, 122, 16.2]],
    'Patient_16_DSA': [[64, 97, 158, 5.5]],
    'Patient_17_DSA': [[96, 79, 143, 3.9]],
    'Patient_18_DSA': [[107, 82, 131, 7.2]],
    'Patient_19_DSA': [[126, 121, 122, 9.5]],
    'Patient_19_DSA2': [[65, 91, 118, 4.2]],
    'Patient_20_DSA': [[91, 76, 158, 7.0]],
    'Patient_21_DSA': [[87, 102, 126, 18.5]],
    'PatientC02_DSA': [[81, 92, 98, 15.5]],
    'PatientC03_DSA': [[64, 81, 131, 4.8]],
    'PatientC05_DSA': [[76, 69, 140, 5.3]],
    'PatientC06_DSA': [[144, 117, 122, 2.3], [141, 108, 124, 2.2]],
    'PatientC07_DSA': [[83, 77, 149, 8.1] , [91, 80, 150, 5.7]],
    'PatientC08_DSA': [[7, 150, 146, 5.5]],
    'PatientC09_DSA': [[79, 76, 132, 2.4]],
    'PatientC10_DSA': [[92, 91, 140, 4.6]],
    'PatientC11_DSA': [[112, 107, 104, 2.2]]
}


def import_dicom(file):
    ds = pydicom.dcmread(os.getcwd() + "/data/" + file)
    file = file.replace(".dcm", "")     
    
    
    #shear = preprocessing.augment.normalize_grayscale(ds.pixel_array)
    #visualize_mask(shear)
    #ds.PixelData = shear
    #ds.save_as(os.getcwd() + "/data/normalized.dcm")

    if file in aneurysm_coordinates.keys():
        ac = aneurysm_coordinates[file]
    else:
        ac = []
    
    return Dicom(file, ac , ds.pixel_array)


def visualize_mask(mask):
    plt.imshow(mask[100],cmap='Greys')
    plt.show()
    plt.imshow(mask[20])
    plt.show()
    plt.imshow(mask[40])
    plt.show()
