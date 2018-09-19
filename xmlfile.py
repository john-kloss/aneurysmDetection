import xml.etree.ElementTree as ET
import numpy as np
import scipy.stats as stats

tree = ET.parse('./data/all_MRA.xml')
root = tree.getroot()
sizewith = []
size = []
for aneurysms in root:
    #string = ("'"+aneurysms.find('name').text+"':"+" [["+aneurysms.find('z').text+", " + aneurysms.find('y').text + ", " 
    #            + aneurysms.find('x').text + ", " + aneurysms.find('size').text + "]],")
    if aneurysms.find('name').text in ["Patient_C127_TOF-MRA","Patient_C38_MRA","Patient_C138_TOF-MRA","Patient_C120_TOF-MRA"]:
        size.append(float(aneurysms.find('size').text))
    if not aneurysms.find('name').text in ["Patient_C127_TOF-MRA","Patient_C38_MRA","Patient_C138_TOF-MRA","Patient_C120_TOF-MRA"]:
        sizewith.append(float(aneurysms.find('size').text))


size= np.array(size)
sizewith= np.array(sizewith)
avg1 = np.mean(size)
std1 = np.std(size)
avg2 = np.mean(sizewith)
std2 = np.std(sizewith)

stat, p = stats.ttest_ind_from_stats(avg1,std1, size.shape[0], avg2,std2, sizewith.shape[0])

avg = np.mean(sizewith)
std = np.std(size)
print(avg, std, np.max(sizewith))
        
    