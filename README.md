## Code-Implementation-of-Super-Resolution-ZOO

Code-Implementation-of-Super-Resolution-ZOO





## Algorithms List

One algorithm have three states：

1. Coding...
2. Code completed, result has not yet been evaluated
3. Result has been evaluated

| Algorithms and Paper URL                       | state | Remarks                      |
| ---------------------------------------------- | ----- | ---------------------------- |
| [s-LWSR](https://arxiv.org/pdf/1909.10774.pdf) | 2     | lightweight(unet / mobilenet similar)                  |
| [FRVSR](https://arxiv.org/pdf/1801.04590.pdf)  | 2     | video / recurrent training   |                    |
| [TeCoGAN](https://arxiv.org/abs/1811.09393v3)  | 1     | video / adversarial training |
| ...                                            |       |                              |



For an algorithm, you can see `"./models/algoname_model.py"` file for training and test scripts details.

e.g. for s-LWSR algorithm,  The beginning of `"./models/lwsr_model.py"` file is as follows:

![training and test scripts](https://s2.ax1x.com/2020/02/06/1cAy6I.png)



## Code Structure

...





## Acknowledgement

This code is built on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) , thank the authors for sharing their codes.




## Datasets and Benchmarks
Please see [HERE](https://github.com/Feynman1999/Code-Implementation-of-Super-Resolution-ZOO/blob/master/datasets-and-benchmark) for the popular SR datasets and certain benchmark results.



## Image & Video Quality Assessment
Please see [HERE](https://github.com/Feynman1999/Code-Implementation-of-Super-Resolution-ZOO/blob/master/iqa) for the Image & Video Quality Assessment methods.




## Super Resolution Challenges

#### 2020

* [New Trends in Image Restoration and Enhancement workshop and challenges on image and video restoration and enhancement in conjunction with CVPR 2020](http://www.vision.ee.ethz.ch/ntire20/)

#### 2019

* [Advances in Image Manipulation workshop and challenges on image and video manipulation in conjunction with ICCV 2019](http://www.vision.ee.ethz.ch/en/aim19/)

* [2019 Alibaba Youku video enhancement and super resolution challenge](https://tianchi.aliyun.com/competition/entrance/231711/introduction)
* [New Trends in Image Restoration and Enhancement workshop and challenges on image and video restoration and enhancement in conjunction with CVPR 2019](http://www.vision.ee.ethz.ch/ntire19/)

#### 2018

* [New Trends in Image Restoration and Enhancement workshop and challenges on super-resolution, dehazing, and spectral reconstruction in conjunction with CVPR 2018](http://www.vision.ee.ethz.ch/ntire18/)


## Recent overview papers

* [Deep Learning on Image Denoising: An overview](https://arxiv.org/abs/1912.13171)

* [Deep Learning for Image Super-resolution: A Survey](https://arxiv.org/pdf/1902.06068.pdf)

* [A Deep Journey into Super-resolution: A Survey](https://arxiv.org/pdf/1904.07523.pdf)
