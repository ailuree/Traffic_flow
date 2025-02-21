# utils/get_roi.py
import cv2
import numpy as np

points = []  # 存储点击的坐标
img = None   # 原始图像
img_show = None  # 显示图像

def mouse_callback(event, x, y, flags, param):
    global points, img, img_show
    
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        points.append((x, y))
        print(f"点击坐标: ({x}, {y})")
        
        # 每次点击重新复制原图，避免重叠
        img_show = img.copy()
        
        # 绘制所有已点击的点
        for point in points:
            cv2.circle(img_show, point, 3, (0, 255, 0), -1)
        
        # 如果已经有4个点，绘制区域
        if len(points) == 4:
            # 转换为numpy数组
            pts = np.array(points, np.int32)
            # 绘制填充多边形
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            # 添加透明效果
            alpha = 0.4
            img_show = cv2.addWeighted(overlay, alpha, img_show, 1-alpha, 0)
            
            print("\n选择的四个点坐标:")
            for i, point in enumerate(points):
                print(f"Point {i+1}: {point}")
            
            # 保存到文件
            with open('./config/roi_config.txt', 'w') as f:
                f.write(str(points))
                
        cv2.imshow('Select ROI', img_show)

def select_roi(video_path):
    global img, img_show
    cap = cv2.VideoCapture(video_path)
    ret, img = cap.read()
    if not ret:
        print("无法读取视频")
        return
    
    img_show = img.copy()
    cv2.imshow('Select ROI', img_show)
    cv2.setMouseCallback('Select ROI', mouse_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    video_path = "../test3.mp4"  # 替换为你的视频路径
    select_roi(video_path)