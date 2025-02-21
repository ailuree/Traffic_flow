# -*- coding: utf-8 -*-
# @Author : pan
# @Description : Flask后端
# @Date : 2023年7月27日10:46:25

import base64
import cv2
import os
import json
import string
import random
import jwt

import numpy as np
import supervision as sv

import time

from datetime import datetime  # 修改导入方式
from typing import Any
from dotenv import load_dotenv
from PIL import Image
from flask import Flask, request, abort, send_from_directory, jsonify, session
from pprint import pprint
from apscheduler.schedulers.background import BackgroundScheduler

from ultralytics import YOLO

from utils.flask_utils import *

# --------------------------------配置加载


load_dotenv(override=True, dotenv_path='config/end-back.env')

# 服务器配置
HOST_NAME = os.environ['HOST_NAME']
PORT = int(os.environ['PORT'])
TOLERANT_TIME_ERROR = int(os.environ['TOLERANT_TIME_ERROR'])  # 可以容忍的时间戳误差(s)

current_dir = os.getcwd()  # 获取当前文件夹的路径
BEFORE_IMG_PATH = os.path.join(current_dir, 'static', os.environ['BEFORE_IMG_PATH'])  # 拼接目标文件夹路径
AFTER_IMG_PATH = os.path.join(current_dir, 'static', os.environ['AFTER_IMG_PATH'])

# 数据库配置
MYSQL_HOST = os.environ['MYSQL_HOST']             # SQL主机
MYSQL_PORT = os.environ['MYSQL_PORT']             # 连接端口
MYSQL_user = os.environ['MYSQL_user']             # 用户名
MYSQL_password = os.environ['MYSQL_password']     # 密码
MYSQL_db = os.environ['MYSQL_db']                 # 数据库名
MYSQL_charset = os.environ['MYSQL_charset']       # utf8

# 实例化数据库
db = SQLManager(host=MYSQL_HOST, port=eval(MYSQL_PORT), user=MYSQL_user,
				passwd=MYSQL_password, db=MYSQL_db, charset=MYSQL_charset)
# result = db.get_one("SELECT * FROM user WHERE username=%s", ('dzp'))
# pprint(result)
# pprint(result['age'])

# Load a model yolo的全局变量）
model = YOLO("./models/car.pt")  # load a pretrained model (recommended for training)
box_annotator = sv.BoxAnnotator(
    thickness=2
)

app = Flask(__name__, static_folder='static')

# 在执行定时任务时，可能会出现CPU耗尽的情况。
# 这个问题可能出现在任务本身需要大量CPU资源或任务设置了过长的时间间隔导致进程变得不稳定，并且占用了整个CPU。
# 为了避免这个问题，可以使用“APScheduler”库提供的schedulres（定时器）和executors（执行器），可以根据具体需求设置
# 链接：https://www.python100.com/html/85441.html

# 关于：Execution of job "scheduled_function (trigger: interval[0:00:10], next run at: 2023-08-01 12:24:15 CST)" skipped: maximum number of running instances reached (1)
# 1、调整定时任务的频率：增加任务的执行间隔时间，以确保一个任务实例能够在下一个实例开始之前完成。
# 例如，将定时任务的执行间隔从10秒增加到20秒或更长时间。
# 2、增加最大运行实例数量：根据你的需求和系统资源情况，可能可以增加允许同时运行的任务实例的最大数量。
# 这通常需要查看你使用的任务调度器或框架的文档，并根据导进行配置。
scheduler = BackgroundScheduler(
    max_instances=1,  # 限制最大实例数
    coalesce=True     # 合并执行错过的任务
)
# 这里写要定时执行的代码
def scheduled_function():
    print('定时任务启动！')
    select_sql = "SELECT id, threshold, url, is_alarm, mode, location " \
          "FROM monitor WHERE is_alarm='开启'"
    monitor_list = db.get_list(select_sql)
    # 循环执行
    for item in monitor_list:
        pid = int(item['id'])
        threshold = int(item['threshold'])
        mode = item['mode']
        location = item['location']
        source = item['url']

        # 检测流是否存在
        if not check_stream_availability(source):
            print(f'该流拉取失败:{source}')
            return False

        # 根据模式选择不同的参数
        if mode == "快速模式":
            iter_model = iter(
                model.track(source=source, show=False, stream=True, iou=0.3, conf=0.3))
        elif mode == "准确模式":
            iter_model = iter(
                model.track(source=source, show=False, stream=True, iou=0.7, conf=0.7))
        for i in range(2):
            result = next(iter_model)  # 这里是检测的核心
            detections = sv.Detections.from_ultralytics(result)
            if result.boxes.id is None:
                continue
            if len(detections) > threshold:
                # 获取当前时间
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                # 保存处理后的图片
                res_url = save_res_img(result.orig_img, detections, f'alarm.jpg')
                # 写入id
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
                # 构造警报信息
                alarm_description = f'车流量:{len(detections)}'
                # 构建插入语句
                insert_sql = "INSERT INTO alarm (location, description, threshold, photo, pid, create_time, remark) " \
                             "VALUES (%s, %s, %s, %s, %s, %s, %s)"

                db.modify(insert_sql, (location, alarm_description, threshold, res_url, pid, current_time, '无'))
                print('警报已记录！')

# 每5分钟执行一次
scheduler.add_job(
    scheduled_function, 
    'interval', 
    seconds=30,       # 增加间隔时间
    max_instances=1   # 限制单个任务的最大实例数
)
scheduler.start()



# 未登录——请求：
# 1、当用户没有登录时，会话中session的’username‘这一个key没有值为none
# 2、当为空的时候，对于管理后端操作（会被拦截器拦截），并返回一个401

# 登录——请求
# 1、当用户进行登录后，会话中的session为空，然后我们为其设置一个session就好了（session为他的username）
# 2、当用户请求，带着他自己的username——服务器判断：（1）session有没有  （2）session中的value 是否等于 username

# 注销——请求（逻辑）
# 1、当他注销时，前端pinia就清空他的数据，并且发生请求给后端，让session清空
# 2、网站跳转到首页

# 拦截器
# 1、当他再次访问的不是白名单时， 判断session中有没有username
# session设置
app.config['SECRET_KEY'] = 'my-secret-key'  # 设置密钥
app.config['PERMANENT_SESSION_LIFETIME'] = 15 * 60 # session时间: 5分钟
# 拦截器白名单
whitelist = ['/', '/login', '/photo', '/recognize', '/video', '/stream']
# 拦截器 （测试前就注释掉！）
@app.before_request
def interceptor():
    if request.path.startswith('/static/'):  # 如果请求路径以 /static/ 开头，则放行
        return
    if request.path in whitelist:  # 白名单放行
        return
    if not session.get('username'):  # 检查是否已登录
        return wrap_unauthorized_return_value('Unauthorized')  # 返回 401 未授权状态码


# 添加header解决跨域
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With, Authorization'
    # 处理OPTIONS预检请求
    if request.method == 'OPTIONS':
        response.headers['Access-Control-Max-Age'] = '1728000'
        return response
    return response

@app.route("/")
def start_server():
    return "欢迎使用交通路况分析系统！后端启动成功！(*^▽^*)"

# JWT，即 JSON Web Token ——这里准备使用TWJ的，但是无奈，不想用 Redis，就算了  java就是sa-token
@app.route('/login', methods=["POST"])
def login():
    try:
        data = request.json  # 获取 JSON 格式的数据
        username = data.get('username').strip()
        password = data.get('password').strip()
        user_info = db.get_one("SELECT * FROM user WHERE username=%s", (username))
        if user_info and user_info['password'] == password:
            session['username'] = username  # 存储session
            return wrap_ok_return_value({'id':user_info['id'],
                                         'avatar':user_info['avatar'],
                                         'username':user_info['username']})
        return wrap_error_return_value('错误的用户名或密码！') # 登陆失败
    # 登陆失败
    except:
        return wrap_error_return_value('系统繁忙，请稍后再试！')

@app.route('/logOut', methods=["get"])
def log_out():
    session.clear()
    return wrap_ok_return_value('账号已退出！')

@app.route('/submitMonitorForm', methods=["POST"])
def submit_monitor_form():
    try:
        data = request.json  # 获取 JSON 格式的数据
        threshold = int(data.get('threshold'))
        person = data.get('person')
        video = data.get('video')
        url = data.get('url')
        if(data.get('is_alarm')):
            is_alarm = '开启'
        else:
            is_alarm = '关闭'
        mode = data.get('mode')
        location = data.get('location')
        create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        remark = data.get('remark')
        # 插入
        insert_sql = "INSERT INTO monitor " \
              "(threshold, person, video, url, is_alarm, mode, location, create_time, create_by, remark) " \
              "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        values = (threshold, person, video, url, is_alarm, mode, location, create_time, "", remark)
        # pprint(values)
        db.modify(insert_sql, values)
        return wrap_ok_return_value('配置提交成功！')
    # 处理异常情况
    except Exception as e:
        pprint(e)
        return wrap_error_return_value('系统繁忙，请稍后再试！')


@app.route('/updateMonitorForm', methods=["POST"])
def update_monitor_form():
    try:
        data = request.json  # 获取 JSON 格式的数据
        id = data.get('id')
        threshold = int(data.get('threshold'))
        person = data.get('person')
        video = data.get('video')
        url = data.get('url')
        if(data.get('is_alarm')):
            is_alarm = '开启'
        else:
            is_alarm = '关闭'
        mode = data.get('mode')
        location = data.get('location')
        remark = data.get('remark')

        # 更新
        update_sql = "UPDATE monitor SET " \
                     "threshold = %s, person = %s, video = %s, url = %s, " \
                     "is_alarm = %s, mode = %s, location = %s, remark = %s " \
                     "WHERE id = %s"
        values = (threshold, person, video, url, is_alarm, mode, location, remark, id)
        db.modify(update_sql, values)

        return wrap_ok_return_value('配置更新成功！')

    except Exception as e:
        return wrap_error_return_value(str(e))

# 查询用户信息（分页查询）
@app.route('/usersList/<int:page>', methods=['GET'])
def get_user_list(page):
    page_from = int((page - 1) * 10)
    page_to = int(page)*10
    select_sql = f"select id, username, avatar, email, grade from user limit {page_from}, {page_to}"
    user_list = db.get_list(select_sql)
    # pprint(user_list)
    return wrap_ok_return_value(user_list)

# 查询监控信息（分页查询）
@app.route('/monitorList/<int:page>', methods=['GET'])
def get_monitor_list(page):
    page_from = int((page - 1) * 10)
    page_to = int(page)*10
    select_sql = f"SELECT id, threshold, person, video, url, is_alarm, mode, " \
          f"location, create_time, create_by, remark FROM monitor" \
          f" limit {page_from}, {page_to}"
    monitor_list = db.get_list(select_sql)
    # 将datetime对象格式化为字符串
    for item in monitor_list:
        item['create_time'] = item['create_time'].strftime('%Y-%m-%d %H:%M:%S')
    # pprint(monitor_list)
    return wrap_ok_return_value(monitor_list)

# 查询警报信息（分页查询）
@app.route('/alarmList/<int:page>', methods=['GET'])
def get_alarm_list(page):
    page_from = int((page - 1) * 10)
    page_to = int(page)*10
    select_sql = f"SELECT id, location, description, threshold, photo, pid, create_time, remark " \
                 f"FROM alarm LIMIT {page_from}, {page_to}"
    alarm_list = db.get_list(select_sql)

    # 将datetime对象格式化为字符串
    for item in alarm_list:
        item['create_time'] = item['create_time'].strftime('%Y-%m-%d %H:%M:%S')

    return wrap_ok_return_value(alarm_list)

@app.route("/photo", methods=["POST"])
def recognize_base64():
    photo_data = request.form.get('photo')  # 获取前端传递的 base64 图片数据
    photo_data = photo_data.replace('data:image/png;base64,', '')  # 去掉 base64 编码中的前缀

    # 解码 base64 数据为二进制数据
    image_data = base64.b64decode(photo_data)

    # 保存为文件
    before_img_path = save_img_base64(image_data, path=BEFORE_IMG_PATH)

    # 处理完成后，返回响应
    name = f"{''.join(random.choice(string.ascii_lowercase) for i in range(5))}.png"

    # 返回结果
    return yolo_res(before_img_path=before_img_path, name=name)

@app.route("/recognize", methods=["POST"])
def recognize_photo():
    photo = request.files['file']
    name = photo.filename
    # return "ok"
    # img = Image.open(photo)
    # img = Image.open(photo).convert("RGB")

    img = cv2.imdecode(np.fromstring(photo.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 保存未处理的图片
    before_img_path = save_img(name,img,BEFORE_IMG_PATH)

    # 返回结果
    return yolo_res(before_img_path=before_img_path, name=name)

# yolo 处理图片
def yolo_res(before_img_path, name):
    try:
        # 获取数据源
        img = Image.open(before_img_path)
        iter_model = iter(
            model.track(source=img, show=False))
        result = next(iter_model)

        detections = sv.Detections.from_ultralytics(result)

        if result.boxes.id is None:
            return wrap_ok_return_value('照片中没有目标物体哟！')
            
        # 写入id
        detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        
        # 修改标签格式
        labels = []
        for xyxy, confidence, class_id, tracker_id in zip(
            detections.xyxy,
            detections.confidence,
            detections.class_id,
            detections.tracker_id
        ):
            x1, y1, x2, y2 = map(int, xyxy)
            label = f"ID: {tracker_id} Class: {model.model.names[class_id]} Confidence: {confidence:.2f} Coordinates: {x1} {y1} {x2} {y2}"
            labels.append(label)

        # 保存处理后的图片
        res_img = result.orig_img
        res_url = save_res_img(res_img, detections)

        return wrap_ok_return_value({
            'labels': labels,
            'after_img_path': res_url,
            'total_objects': len(labels)
        })
    except Exception as e:
        print(str(e))
        return wrap_error_return_value('服务器繁忙，请稍后再试！')

def save_res_img(res_img, detections, name = 'default.jpg'):
    # 手动绘制边界框和标签
    img_box = res_img.copy()
    for xyxy, tracker_id in zip(detections.xyxy, detections.tracker_id):
        x1, y1, x2, y2 = map(int, xyxy)
        label = f"ID: {tracker_id}"
        
        # 绘制边界框
        cv2.rectangle(img_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制标签背景
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_box, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
        
        # 绘制标签文本
        cv2.putText(img_box, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 将 BGR 格式的 frame 转换为 RGB 格式
    rgb_frame = cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB)
    numpy_frame = np.array(rgb_frame)
    
    after_img_path = save_img(name, numpy_frame, AFTER_IMG_PATH)
    return after_img_path.replace(current_dir, "http://127.0.0.1:5500/").replace('\\', '/')


# 修改process_frame函数，添加更多视觉信息
def process_frame(frame, detections):
    """处理单帧图像，添加边界框和标签"""
    img_box = frame.copy()
    
    # 添加检测信息到帧
    current_objects = len(detections)
    info_text = f"Objects Detected: {current_objects}"
    cv2.putText(img_box, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    for xyxy, confidence, class_id, tracker_id in zip(
        detections.xyxy,
        detections.confidence,
        detections.class_id,
        detections.tracker_id
    ):
        x1, y1, x2, y2 = map(int, xyxy)
        label = f"ID: {tracker_id} {model.model.names[class_id]} {confidence:.2f}"
        
        # 使用不同颜色标识不同类别
        color = (0, 255, 0)  # 默认绿色
        if class_id == 0:  # 假设0是汽车
            color = (0, 255, 0)
        elif class_id == 1:  # 假设1是卡车
            color = (255, 0, 0)
        
        # 绘制边界框
        cv2.rectangle(img_box, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签背景
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_box, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
        
        # 绘制标签文本
        cv2.putText(img_box, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img_box

# 添加一个新的路由��获取视频处理进度
@app.route("/video/progress/<video_id>", methods=["GET"])
def get_video_progress(video_id):
    # 这里可以实现进度查询逻辑
    # 可以使用Redis或其他存储来保存进度信息
    return wrap_ok_return_value({
        'progress': 50,  # 示例进度
        'status': 'processing'
    })

# 添加一个路由来访问处理后的视频
@app.route("/video/result/<filename>")
def get_video_result(filename):
    try:
        return send_from_directory(AFTER_IMG_PATH, filename)
    except Exception as e:
        return wrap_error_return_value(f'无法访问视频: {str(e)}')

# 添加视频流处理的路由
@app.route("/stream", methods=["POST"])
def process_stream():
    try:
        data = request.json
        stream_url = data.get('url')
        
        if not stream_url:
            return wrap_error_return_value('未提供视频流URL')
            
        # 使用新的检查函数
        if not check_stream_availability(stream_url):
            return wrap_error_return_value('无法访问视频流，请检查URL或网络连接')
            
        # 获取视频信息
        cap = cv2.VideoCapture(stream_url)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 创建输出视频的路径和写入对象
        output_path = os.path.join(AFTER_IMG_PATH, "stream_result.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 处理视频流
        frame_count = 0
        max_frames = 300  # 限制处理的帧数
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 使用YOLO处理帧
            results = model.track(source=frame, show=False)
            
            for result in results:
                detections = sv.Detections.from_ultralytics(result)
                if result.boxes.id is not None:
                    detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
                    processed_frame = process_frame(frame, detections)
                    out.write(processed_frame)
            
            frame_count += 1
        
        # 释放资源
        cap.release()
        out.release()
        
        # 构建结果URL
        result_url = f"http://127.0.0.1:5500/static/{os.environ['AFTER_IMG_PATH']}/stream_result.mp4"
        
        return wrap_ok_return_value({
            'video_url': result_url,
            'processed_frames': frame_count,
            'fps': fps
        })
        
    except Exception as e:
        return wrap_error_return_value(f'视频流处理失败: {str(e)}')

# 在 Flask 中，警告消息"WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead."
# 表示您正在使用开发服务器，这不适合在生产环境中使用。如果您要切换到生产模式，您需要使用一个生产级的 WSGI 服务器。
# 通常使用类似于 Gunicorn、uWSGI 或 Nginx + uWSGI 的组合来部署和运行 Flask 应用。
if __name__ == "__main__":
    # 从命令行参数获取host和port
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5500)
    args = parser.parse_args()
    
    app.run(
        host=args.host,
        port=args.port,
        debug=False,
        # 添加threaded参数支持多线
        threaded=True
    )

def check_stream_availability(stream_url, max_retries=3):
    """检查视频流是否可用，带重试机制"""
    for attempt in range(max_retries):
        try:
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                print(f"尝试 {attempt + 1}/{max_retries} 失败")
                continue
                
            # 尝试读取一帧
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                return True
            
        except Exception as e:
            print(f"尝试 {attempt + 1}/{max_retries} 出错: {str(e)}")
            
        time.sleep(1)  # 等待1秒后重试
    
    print(f'该流拉取失败:{stream_url}')
    return False
