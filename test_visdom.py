# import visdom
# import numpy as np
#
# vis = visdom.Visdom(server="http://localhost", port=8097, env="main")
# vis.video(videofile = 'datasets/dragonball/train/A/1.avi')

import sys
# import y4m
# import os
# import numpy as np
# import cv2
# import time
#
# def process_frame(frame):
#     # do something with the frame
#     if frame.count == 0:
#         print(len(frame.buffer), type(frame.buffer), frame.headers)
#         assert frame.headers['C'] == '420mpeg2', 'wrong codec'
#
#     rows = frame.headers['H']
#     cols = frame.headers['W']
#     rc = rows * cols
#     assert rows % 2 == 0 and cols % 2 == 0, 'otherwise maybe wrong?'
#     img_y = np.reshape(np.fromstring(frame.buffer[:rc], np.uint8), (rows, cols))
#     img_u = np.reshape(np.fromstring(frame.buffer[rc:rc//4 + rc], np.uint8), (rows // 2, cols // 2))
#     img_v = np.reshape(np.fromstring(frame.buffer[rc//4+rc:], np.uint8), (rows // 2, cols // 2))
#     enlarge_u = cv2.resize(img_u, dsize=(cols, rows), interpolation=cv2.INTER_CUBIC)
#     enlarge_v = cv2.resize(img_v, dsize=(cols, rows), interpolation=cv2.INTER_CUBIC)
#     dst = cv2.merge([img_y, enlarge_u, enlarge_v])
#     bgr = cv2.cvtColor(dst, cv2.COLOR_YUV2BGR)
#     print(time.time())
#     # cv2.imshow("dst", bgr)
#     # cv2.waitKey(0)
#
# if __name__ == '__main__':
#     file = "datasets/youku/train/A/"
#     for root, _, fnames in sorted(os.walk(file)):
#         for fname in fnames:
#                 path = os.path.join(root, fname)
#                 parser = y4m.Reader(process_frame, verbose=False)
#
#                 infd = open(path, "rb")
#
#                 with infd as f:
#                     while True:
#                         data = f.read(1024)
#                         if not data:
#                             break
#                         parser.decode(data)

from data.video_folder import read_video
import cv2
import matplotlib.pyplot as plt
import  numpy

path1 = "datasets/youku/train/A/Youku_00031_l.y4m"
path2 = "datasets/dragonball/train/A/0001.mkv"

li = read_video(path1)
w, h = li[0].size

for i in range(len(li)):
    li[i] = numpy.array(li[i])[..., ::-1]
    li[i] = cv2.resize(li[i], (h, w), interpolation=cv2.INTER_CUBIC)

out = cv2.VideoWriter("test.avi", cv2.VideoWriter_fourcc(*'I420'), 25, (h, w))
for frame in li:
    out.write(frame)
out.release()
