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

![training and test scripts](https://s2.ax1x.com/2020/01/10/l4Fejf.png)





## Code Structure

...





## Acknowledgement

This code is built on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) , thank the authors for sharing their codes.


## Datasets and Benchmarks
Please see [HERE](https://github.com/Feynman1999/Code-Implementation-of-Super-Resolution-ZOO/blob/master/datasets-and-benchmark) for the popular SR datasets and certain benchmark results.

## Image & Video Quality Assessment
Please see [HERE](https://github.com/Feynman1999/Code-Implementation-of-Super-Resolution-ZOO/blob/master/iqa) for the Image & Video Quality Assessment methods.


## Super Resolution Challenges

* [2019 Alibaba Youku video enhancement and super resolution challenge](https://tianchi.aliyun.com/competition/entrance/231711/introduction)

