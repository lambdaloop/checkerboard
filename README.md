# Checkerboard

[![PyPI version](https://badge.fury.io/py/checkerboard.svg)](https://badge.fury.io/py/checkerboard)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

I could not find any libraries to find checkerboards robustly in Python, except for OpenCV.
However, OpenCV's checkerboard tends to fail when the checkerboard is somewhat blurred or rotated.

Hence, this library was born. It may be slightly slower than OpenCV, but it will **find** that checkerboard.

Comparison of OpenCV vs this library (OpenCV on the left, checkerboard on the right):
![comparison](https://thumbs.gfycat.com/HomelyAccurateBettong.webp)

## Quickstart

You can install checkerboard easily through pip:
```bash
pip install checkerboard
```

Then you can go ahead and detect checkerboards like so:
```python
from checkerboard import detect_checkerboard

size = (9, 6) # size of checkerboard
image = ... # obtain checkerboard
corners = detect_checkerboard(image, size)
```

## References

The implementation of checkerboard detection is mainly based on [libcbdetect](http://www.cvlibs.net/software/libcbdetect/) 
and accompanying paper:
```
@INPROCEEDINGS{Geiger2012ICRA,
  author = {Andreas Geiger and Frank Moosmann and Oemer Car and Bernhard Schuster},
  title = {Automatic Calibration of Range and Camera Sensors using a single Shot},
  booktitle = {International Conference on Robotics and Automation (ICRA)},
  year = {2012}
} 
```

