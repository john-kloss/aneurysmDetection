import xml.etree.ElementTree as ET

tree = ET.parse('./data/all_DSA.xml')
root = tree.getroot()

for aneurysms in root:
    string = ("'"+aneurysms.find('name').text+"':"+" [["+aneurysms.find('z').text+", " + aneurysms.find('y').text + ", " 
                + aneurysms.find('x').text + ", " + aneurysms.find('size').text + "]],")
    print(string)
        
    