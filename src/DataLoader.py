import os
import xml.etree.ElementTree as ET

def list_files(directory)-> None:
    count: int = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            print(os.path.join(root, file))
            count += 1
    print(f"\nTotal files: {count}")


def count_defect_classes(directory: str) -> None:
    

    pass


def print_xml_recursively(element, indent=0) -> None:
    # Print current element's tag, attributes, and text
    print(
        " " * indent + f"Tag: {element.tag}, Attributes: {element.attrib}, Text: {element.text.strip() if element.text else ''}")

    # Recursively print all child elements
    for child in element:
        print_xml_recursively(child, indent + 2)





