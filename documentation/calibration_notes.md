# Calibration notes

### Testing stereo synchronisation

The vision data transferred from the R1 to the backpack do not contain timestamps. We need each pair of images to be synchronised, because otherwise the disparity map becomes unreliable. To test how much delay there is between the pairs, we will make use of a digital clock:

| Left View | Right View |
| :---: | :---: |
| ![Left View 0](skew/left_view_0.png) | ![Right View 0](skew/right_view_0.png) |
| ![Left View 1](skew/left_view_1.png) | ![Right View 1](skew/right_view_1.png) |
| ![Left View 2](skew/left_view_2.png) | ![Right View 2](skew/right_view_2.png) |
| ![Left View 3](skew/left_view_3.png) | ![Right View 3](skew/right_view_3.png) |
| ![Left View 4](skew/left_view_4.png) | ![Right View 4](skew/right_view_4.png) |
| ![Left View 5](skew/left_view_5.png) | ![Right View 5](skew/right_view_5.png) |
| ![Left View 6](skew/left_view_6.png) | ![Right View 6](skew/right_view_6.png) |
| ![Left View 7](skew/left_view_7.png) | ![Right View 7](skew/right_view_7.png) |
| ![Left View 8](skew/left_view_8.png) | ![Right View 8](skew/right_view_8.png) |
| ![Left View 9](skew/left_view_9.png) | ![Right View 9](skew/right_view_9.png) |

Since the stereo camera is capturing at 10Hz, while the clock is updating at >100Hz, these results indicate that the skew is extremely low (if there is any skew). 

### Testing results from disparity model

The Unitree R1 EDU does not provide any instrinsic camera parameters, so we need to do so ourselves. After calibrating, here are the results:

~75cm away:

![Depth 75](depth/75.png)

~125cm away:

![Depth 125](depth/125.png)

~100cm away:

![Depth 100](depth/100.png)

Note the depth here means the forward-axis depth $Z$, so a flat wall 1m away from the camera should have approximately 1m everywhere in the depth image. Overall, these results show that the calibration gives a fairly good result. However these are with approximate distances (using a 40cm ruler), and would require a more controlled environment to do any kind of comprehensive test.