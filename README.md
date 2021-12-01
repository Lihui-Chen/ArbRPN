# ArbRPN
This is the official implementation of 'ArbRPN: A Bidirectional Recurrent Pansharpening Network for Multispectral Images with Arbitrary Numbers of Bands', Accepted by IEEE Transaction on Geoscience and Remote Sensing, [[DOI:10.1109/TGRS.2021.3131228]](https://ieeexplore.ieee.org/document/9627886)
# Requirements
Python 3.6

Pytorch >= 1.1

torchvision

pandas

PIL

opencv-python

numpy

random

scipy

importlib

# Quick Test
python test.py -net_arch ArbRPN -opt options/test/test.yml -trained_model models/ArbRPN_QB_MIX4.pth

# Training
To do.

# Test
To do.


If you find these codes are helpful, please kindly cite

```latex
@ARTICLE{9627886,
  author={Chen, Lihui and Lai, Zhibing and Vivone, Gemine and Jeon, Gwanggil and Chanussot, Jocelyn and Yang, Xiaomin},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={ArbRPN: A Bidirectional Recurrent Pansharpening Network for Multispectral Images with Arbitrary Numbers of Bands}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TGRS.2021.3131228}}
```





