这个项目的目的是构建一个适用于图片/视频超分辨的
train/test代码pipeline


todolist:

* video可视化时 三张视频拼在一起 方便对比

* 检查参数初始化是否有问题，优化初始化的流程

* 设置随机数种子

* 修改 base networks  将多余的放到Pix2pix   仔细思考 什么是多余的

* 像商业软件那样 拖动中间竖条 可以对比两张图片的前端

* visdom显示视频

* dataloader 加速！

* bug : test的时候打印options没有hard code的内容 思考如何修改

* 仔细思考 Options   四个部分整理一下

* 分布式训练

* continue_train   adam?

## help

first run 

`·python -m visdom.server`



## idea

* 每次batchsize = 1  但是crop的size不一样 这样会不会效果好点


