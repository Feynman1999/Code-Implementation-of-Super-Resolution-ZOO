这个项目的目的是构建一个适用于图片/视频超分辨的
train/test代码pipeline


todolist:

* 完成frvsr

* 完成iqa

* 检查参数初始化是否有问题

* 设置随机数种子

* 重新调整options的位置及其结构

* 修改 base networks  将多余的放到Pix2pix   仔细思考 什么是多余的

* 像商业软件那样 拖动中间竖条 可以对比两张图片的前端

* visdom显示视频

* dataloader 加速！


## help

first run 

`·python -m visdom.server`



## idea

* 每次batchsize = 1  但是crop的size不一样 这样会不会效果好点


