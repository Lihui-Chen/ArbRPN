# ArbRPN
This is the official implementation of 'ArbRPN: A Bidirectional Recurrent Pansharpening Network for Multispectral Images with Arbitrary Numbers of Bands', Accepted by IEEE Transaction on Geoscience and Remote Sensing, [[DOI:10.1109/TGRS.2021.3131228]](https://ieeexplore.ieee.org/document/9627886)
# News
The bug of the Q2n metric is fixed by the codes from the following paper.
``` latex
@ARTICLE{9447896,
  author={Vivone, Gemine and Dalla Mura, Mauro and Garzelli, Andrea and Pacifici, Fabio},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={A Benchmarking Protocol for Pansharpening: Dataset, Preprocessing, and Quality Assessment}, 
  year={2021},
  volume={14},
  number={},
  pages={6102-6118},
  doi={10.1109/JSTARS.2021.3086877}}
```

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
  year={2022},
  volume={60},
  number={},
  pages={1-18},
  doi={10.1109/TGRS.2021.3131228}}
```





