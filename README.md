# Binarized Convolutional Landmark Localizers for Real-time Human Pose Estimation and Face Alignment

This code implements a demo of the Binarized Convolutional Landmark Localizers for Real-time Human Pose Estimation and Face Alignment paper by Adrian Bulat and Georgios Tzimiropoulos.

*** Training code will be released soon ***

## Requirements
- Install the latest [Torch7](http://torch.ch/docs/getting-started.html) version (for Windows, please follow the instructions avaialable [here](https://github.com/torch/distro/blob/master/win-files/README.md))

### Packages
- [cutorch](https://github.com/torch/cutorch)
- [nn](https://github.com/torch/nn)
- [cudnn](https://github.com/soumith/cudnn.torch) (cudnn5 preffered)
- [xlua](https://github.com/torch/xlua)
- [image](https://github.com/torch/image)
- [gnuplot](https://github.com/torch/gnuplot)
- [cURL](https://github.com/Lua-cURL/Lua-cURLv3)
- [paths](https://github.com/torch/paths)

## Setup
Clone the github repository
```bash
git clone https://github.com/1adrianb/binary-human-pose-estimation --recursive
cd binary-human-pose-estimation
```

Build and install the BinaryConvolution package
```bash
cd bnn.torch/; luarocks make; cd ..;
```

Install the modified optnet package
```bash
cd optimize-net/; luarocks make; cd ..;
```

Run the following command to prepare the files required by the demo. This will download 10 images from the MPII dataset alongside the dataset structure converted to .t7
```bash
th download-content.lua
```

## Usage

In order to run the demo simply type:
```bash
th main.lua
```

## Pretrained models

## Notes

For more details/questions please visit the [project page](https://www.adrianbulat.com/binary-points-localisation) or send an email at adrian.bulat@nottingham.ac.uk




