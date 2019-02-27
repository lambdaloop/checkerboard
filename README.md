# Checkerboard

I could not find any libraries to find checkerboards robustly in Python, except for OpenCV.
However, OpenCV's checkerboard tends to fail when the checkerboard is somewhat blurred or rotated.

Hence, this library was born. It may be slightly slower than OpenCV, but it will **find** that checkerboard.

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

