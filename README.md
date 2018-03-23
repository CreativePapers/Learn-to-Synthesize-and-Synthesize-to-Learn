# ECCV2018
This repository includes the training and testing codes for ECCV 2018 submission: "Learn to Synthesize and Synthesize to Learn".

![](https://github.com/CreativePapers/ECCV2018/blob/master/146_fake.png)

### Dependencies
- [Python2.7](https://www.anaconda.com/download/#linux)
- [PyTorch](http://pytorch.org/)
- [torchvision](http://pytorch.org/docs/master/torchvision)
- [OpenCV](https://opencv.org/)
- [Dlib](http://dlib.net/)

### Datasets
- [The Binghamton University 3D Facial Expression Database (BU-3DFE)](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html)
- [The Radboud Faces Database (RaFD)](http://www.socsci.ru.nl:8180/RaFD2/RaFD?p=main)
- [The MUG dataset](https://mug.ee.auth.gr/fed/)
- [Oulu-CASIA VIS](http://www.cse.oulu.fi/CMV/Downloads/Oulu-CASIA)

## Example Usage for expression synthesis model:

### Clone the repository
```
$ git clone https://github.com/CreativePapers/ECCV2018.git
cd ECCV2018-master
```
### Train
```
python expression_synthesis.py --mode='train'
```
### Test
```
python expression_synthesis.py --mode='test'
```
### Attribute-Guided Face Synthesis Model

![](https://github.com/CreativePapers/ECCV2018/blob/master/diagram.pdf)

### Demo

[![Watch the demo](https://github.com/CreativePapers/ECCV2018/blob/master/supplementary.mp4)

### Sample Results 

![](https://github.com/CreativePapers/ECCV2018/blob/master/2_fake.png)

![](https://github.com/CreativePapers/ECCV2018/blob/master/10_fake.png)

## TO DO
To add pose normalization model




