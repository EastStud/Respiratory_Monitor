orig_only.py is meant to be run to collect sample data for analysis.

Obviously, data must exist befor analyzing.  orig_only.py will save a pickle 'pkl'
file in the current working directory.

Once pickle files exist, they can be referenced in the command line when
executing Analysis_5.py.

There is a help '-h' option for Analysis_5.py for additional arguments to be passed.

As far as running the Kinect with the code, there are some requirements listed below:

Software:
1) Linux OS (I specifically ran a Ubuntu 64-bit VM)
2) KinectSDK 1.0 if using the XBOX 360 Kinect otherwise use the 2.0 SDK
3) Python and Python libraries to include: freenect, Cython, Matplotlib, Scipy, OpenCV

Hardware:
1) Microsoft XBOX 360 Kinect Sensor
2) Microsoft Kinect PC AC Adaptor

Note: That you should test the Kinect is running before running orig_only.py.
Sometimes you'll get unwanted behavior.

Also be aware that the pickle files can be very large.  A 60 second set of samples
will be roughly 1 GB in size.