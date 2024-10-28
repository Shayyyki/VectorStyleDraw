import os
import cv2

file_dir = './content/res/'
list = []
canvas_width, canvas_height = 600, 926
# list.sort(key=lambda x:int(x.split('.')[0]))
for root , dirs, files in os.walk(file_dir):
    for file in files:
        list.append(file)      # 获取目录下文件名列表
list.sort(key=lambda x:int(x[5:-4]))#排序
# VideoWriter是cv2库提供的视频保存方法，将合成的视频保存到该路径中
video = cv2.VideoWriter('./face9.avi', cv2.VideoWriter_fourcc(*'MJPG'), 15, (canvas_width, canvas_height))
for i in range(1, len(list)):
    # print('./content/res/' + list[i-1])
    img = cv2.imread('./content/res/' + list[i-1])
# resize方法是cv2库提供的更改像素大小的方法
# 将图片转换为1280*720像素大小
    img = cv2.resize(img, (canvas_width, canvas_height))
    # 写入视频
    video.write(img)

# 释放资源
video.release()