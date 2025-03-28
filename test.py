from ultralytics import YOLO
import cv2
import os
import pandas as pd

def predict():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    weights_path = os.path.join(current_dir, 'output',  'fovea_detection', 'weights', 'best.pt')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"找不到模型:{weights_path}\n")
    model = YOLO(weights_path)
    
    results = []
    test_dir = os.path.join(current_dir, 'data', 'detection', 'test')
    
    #处理测试图像
    for img_file in sorted(os.listdir(test_dir)):
        img_path = os.path.join(test_dir, img_file)
        pred = model(img_path)[0]#返回的是result对象列表
        
        img = cv2.imread(img_path)#返回(height, width, channels)
        height, width = img.shape[:2]
        
        # 获取预测框的中心点
        if len(pred.boxes) > 0:
            box = pred.boxes[0]#返回第一个预测框，可信度最高
            x_center = float(box.xywh[0][0])
            y_center = float(box.xywh[0][1])
        else:
            x_center = width / 2#预测失败用图像最中心点代替
            y_center = height / 2
        
        # 添加结果
        img_id = img_file.split('.')[0]
        img_id = str(int(img_id))
        results.extend([
            {'ImageID': f'{img_id}_Fovea_X', 'value': x_center},
            {'ImageID': f'{img_id}_Fovea_Y', 'value': y_center}
        ])
    
    # 保存结果
    submission_path = os.path.join(current_dir, 'submission.csv')
    df = pd.DataFrame(results)#用DataFrame对象存结果
    df.to_csv(submission_path, index=False)#保存为CSV文件
    print(f"预测结果已保存到: {submission_path}")

if __name__ == '__main__':
    predict() 