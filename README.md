# Checkerboard

[![PyPI version](https://badge.fury.io/py/checkerboard.svg)](https://badge.fury.io/py/checkerboard)
[![GitHub license](https://img.shields.io/github/license/lambdaloop/checkerboard.svg)](https://github.com/lambdaloop/checkerboard/blob/master/LICENSE)

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
corners, score = detect_checkerboard(image, size)
```

The `corners` returned are in the same format as the
`findChessboardCorners` function from OpenCV, and are already computed
to subpixel precision.

The `score` returned is a metric of the quality of the checkerboard
detection. A perfectly detected checkerboard would have a score of 0,
whereas a bad detection would have a score of 1.


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

