import src.DataLoader as dl
import xml.etree.ElementTree as ET

dl.count_defect_classes(r'kaggle\input\codebrim-original\original_dataset\annotations\image_0000005.xml')


# Load and parse the XML file
tree = ET.parse(r'kaggle\input\codebrim-original\original_dataset\annotations\image_0000005.xml')  # Replace with your XML file path
root = tree.getroot()

# Print the entire file recursively
dl.print_xml_recursively(root)
print()