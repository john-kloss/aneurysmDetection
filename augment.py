from scipy.ndimage import rotate
import pydicom
import numpy as np
import input

NUM_ROTATE = 10

def rotate_image(dicoms):

    rotations = []
    coordinates = []
    for i in range(len(dicoms)):
        dicom = dicoms[i]
        for n in range(NUM_ROTATE):
            rotations.append(rotate(dicom.pixel_array, axes=(0,1), angle=(np.random.randint(1,180)), reshape=False))
            #coordinates.append(TODO)
            
        rotated_data = {
            str(i) : { "images" : [dicom], "coordinates" : dicom.aneurysm },
            "rotations" : { "images" : rotations, "coordinates" :  [0,0,0] }
        }
        
        # ds.PixelData=rotation
        # ds.save_as(os.getcwd()+"/data/tmp/test.dcm")
