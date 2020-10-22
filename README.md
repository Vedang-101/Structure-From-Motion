# Structure from Motion (Major Project 2020-21)
Created a structure from motion framework using OpenCV and Python 3.6.5 referring to Hartley and Zisserman from their seminal book Multiple View Geometry in Computer
Vision. The Project attempts to understand the process of capturing sequence of images and reconstructing the geometry from the images with help of epipolar lines and Triangulation

# Project Content
The project will include a main.py file which is the source of the program, it is programmed to run two pair of images present in Resoures/Images folder. There exists a file named PLY.py containing a class to write .ply object files and storing them.<br>
Along with these, the project also includes a Images folder which includes a sample image sequence which ca be passed to main.py for execution.

# Dependencies
To sucessfully run the program following modules need to be installed
- OpenCV 3.4.2
- OpenCV-Contib (xfeatures2d for SIFT/SURF)
- numpy 1.16.4

To install all these modules you can use pip
```
pip install opencv-python==3.4.2.16
```
```
pip install opencv-contrib-python==3.4.2.16
```
```
pip install numpy==1.16.4 
```

# How to run
To run the program just run main.py making sure the image pair is present in Resources/Images folder with names 0000.jpg and 0001.jpg respectively. Also make sure all dependencies required are installed
