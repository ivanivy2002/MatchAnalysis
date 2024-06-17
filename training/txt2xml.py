import os
import xml.etree.ElementTree as ET

def txt_to_xml(txt_file_path, xml_file_path, categories):
    """
    将txt文件转换为xml文件。
    
    参数:
    txt_file_path (str): 输入的txt文件路径
    xml_file_path (str): 输出的xml文件路径
    categories (list): 类别名称列表
    """
    with open(txt_file_path, 'r') as f:
        lines = f.readlines()
    
    objects = [line.strip().split() for line in lines]
    
    annotation = ET.Element('annotation')
    
    for obj in objects:
        obj_id, x_center, y_center, width, height = map(float, obj)
        obj_id = int(obj_id)
        
        object_element = ET.SubElement(annotation, 'object')
        name_element = ET.SubElement(object_element, 'name')
        name_element.text = categories[obj_id]
        
        bndbox = ET.SubElement(object_element, 'bndbox')
        
        xmin = x_center - width / 2
        ymin = y_center - height / 2
        xmax = x_center + width / 2
        ymax = y_center + height / 2
        
        xmin_element = ET.SubElement(bndbox, 'xmin')
        xmin_element.text = str(xmin)
        ymin_element = ET.SubElement(bndbox, 'ymin')
        ymin_element.text = str(ymin)
        xmax_element = ET.SubElement(bndbox, 'xmax')
        xmax_element.text = str(xmax)
        ymax_element = ET.SubElement(bndbox, 'ymax')
        ymax_element.text = str(ymax)
    
    xml_data = ET.tostring(annotation, encoding='unicode', method='xml')
    
    with open(xml_file_path, 'w') as f:
        f.write(xml_data)

def batch_convert_txt_to_xml(input_dir, output_dir, categories):
    """
    批量将指定目录中的txt文件转换为xml文件。
    
    参数:
    input_dir (str): 输入txt文件的目录
    output_dir (str): 输出xml文件的目录
    categories (list): 类别名称列表
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.txt'):
            txt_file_path = os.path.join(input_dir, file_name)
            xml_file_path = os.path.join(output_dir, file_name.replace('.txt', '.xml'))
            txt_to_xml(txt_file_path, xml_file_path, categories)

# 示例用法
# 转到该文件所在目录
os.chdir(os.path.dirname(__file__))
print(os.getcwd())
categories = ['ball', 'goalkeeper', 'player', 'referee']
input_dir = './football-players-detection-9'
modes = ['train', 'val', 'test']
output_dir = 'annotated'
batch_convert_txt_to_xml(f'{input_dir}/train/labels', f'{output_dir}/train/labels', categories)
batch_convert_txt_to_xml(f'{input_dir}/val/labels', f'{output_dir}/val/labels', categories)
batch_convert_txt_to_xml(f'{input_dir}/test/labels', f'{output_dir}/test/labels', categories)
