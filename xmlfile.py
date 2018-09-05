import xml.etree.ElementTree as ET
import numpy as np

tree = ET.parse('./data/all_MRA.xml')
root = tree.getroot()

size = []
for aneurysms in root:
    #string = ("'"+aneurysms.find('name').text+"':"+" [["+aneurysms.find('z').text+", " + aneurysms.find('y').text + ", " 
    #            + aneurysms.find('x').text + ", " + aneurysms.find('size').text + "]],")
    size.append(float(aneurysms.find('size').text))
avg = np.mean(size)
std = np.std(size)
print(avg, std, np.max(size))
        
    