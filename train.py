from ultralytics import YOLO
import os
import xml.etree.ElementTree as ET
import numpy as np
import shutil
import torch

def convert_xml_to_yolo(xml_path, img_width, img_height):#标注转换为yolo格式
    tree = ET.parse(xml_path)
    root = tree.getroot()

    bbox = root.find('.//bndbox')#标注框
    xmin = float(bbox.find('xmin').text)
    ymin = float(bbox.find('ymin').text)
    xmax = float(bbox.find('xmax').text)
    ymax = float(bbox.find('ymax').text)
    
    x_center = ((xmin + xmax) / 2) / img_width#归一化
    y_center = ((ymin + ymax) / 2) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    
    return [0, x_center, y_center, width, height]#0表示类别

def prepare_dataset():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    dataset_dir = os.path.join(current_dir, 'dataset')#yolo规定数据集目录结构
    label_dir = os.path.join(dataset_dir, 'labels', 'train')
    train_dir = os.path.join(dataset_dir, 'images', 'train')
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    xml_dir = os.path.join(current_dir, 'data', 'detection', 'train_location')
    img_dir = os.path.join(current_dir, 'data', 'detection', 'train')
    
    for xml_file in os.listdir(xml_dir):
        xml_path = os.path.join(xml_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        yolo_bbox = convert_xml_to_yolo(xml_path, width, height)
        
        #保存标注
        img_id = xml_file.split('.')[0]
        label_path = os.path.join(label_dir, f'{img_id}.txt')
        with open(label_path, 'w') as f:
            f.write(' '.join(map(str, yolo_bbox)))#每行一个标注框信息
        
        img_file = f'{img_id}.jpg'
        detection_img_path = os.path.join(img_dir, img_file)
        dataset_img_path = os.path.join(train_dir, img_file)
        
        if os.path.exists(detection_img_path):
            shutil.copy2(detection_img_path, dataset_img_path)#把原来给出的训练集图像转到yolo指定的数据集目录
    
    dataset_yaml = os.path.join(dataset_dir, 'dataset.yaml')#yolo规定数据集配置文件
    with open(dataset_yaml, 'w') as f:
        f.write(
            f'''
            path: {dataset_dir}
            train: images/train
            val: images/train 
            nc: 1#只有一个类别
            names: ['Fovea']
            '''
        )
    
    return dataset_yaml

def train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    dataset_yaml = prepare_dataset()
    
    weights_dir = os.path.join(current_dir, 'weights')#模型
    os.makedirs(weights_dir, exist_ok=True)

    project_dir = os.path.join(current_dir, 'output')#结果
    os.makedirs(project_dir, exist_ok=True)
    
    model_name = 'yolov8n.pt'
    pretrained_model = os.path.join(weights_dir, model_name)
    
    # 加载模型
    if os.path.exists(pretrained_model):
        print(f"加载本地预训练模型: {pretrained_model}")
        model = YOLO(pretrained_model)
    else:
        print(f"下载{model_name}模型中。")
        model = YOLO(model_name)#如果加载不成功会自动下载并加载模型
        shutil.move(model_name, pretrained_model)
        
    model.data_augmentation = {
        'hflip': True,  
        'vflip': True,  
        'rotate': 15, 
        'scale': (0.8, 1.2), 
        'color': (0.5, 1.5),  
    }
    
    # 训练模型
    model.train(
        data=dataset_yaml,          
        epochs=100,                
        imgsz=640,              #图像尺寸
        batch=16,               #批量大小
        name='fovea_detection',    
        project=project_dir,       
        exist_ok=True,            
        optimizer='AdamW',        
        lr0=0.001,              #学习率
    )
if __name__ == '__main__':
    train()
