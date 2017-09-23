# Structure-From-Motion
Basic 3D reconstruction algorithm

## Input video images
The pipeline is as follows:
1. Get good features to track using ShiTomasi corner detection. This is used only once as initialization.
2. Track the initialized points over the entire video.
3. Process the list of points.

Note: I don't know what happens if the points are lost during tracking. In this video the number of points is constant.

![demo](media/out.gif)

## Visualization of the 3D reconstruction.
The number of points in this plot is the same as the number of points that have been tracked.

![demo](media/out-1.gif)
