# Grab_MultipleCameras.cpp
# ============================================================================
# This sample illustrates how to grab and process images from multiple cameras
# using the CInstantCameraArray class. The CInstantCameraArray class represents
# an array of instant camera objects. It provides almost the same interface
# as the instant camera for grabbing.
# The main purpose of the CInstantCameraArray is to simplify waiting for images and
# camera events of multiple cameras in one thread. This is done by providing a single
# RetrieveResult method for all cameras in the array.
# Alternatively, the grabbing can be started using the internal grab loop threads
# of all cameras in the CInstantCameraArray. The grabbed images can then be processed by one or more
# image event handlers. Please note that this is not shown in this example.
# ============================================================================

import os

os.environ["PYLON_CAMEMU"] = "3"

from pypylon import genicam
from pypylon import pylon
import sys
import numpy as np
import cv2

# Number of images to be grabbed.
countOfImagesToGrab = 10

# Limits the amount of cameras used for grabbing.
# It is important to manage the available bandwidth when grabbing with multiple cameras.
# This applies, for instance, if two GigE cameras are connected to the same network adapter via a switch.
# To manage the bandwidth, the GevSCPD interpacket delay parameter and the GevSCFTD transmission delay
# parameter can be set for each GigE camera device.
# The "Controlling Packet Transmission Timing with the Interpacket and Frame Transmission Delays on Basler GigE Vision Cameras"
# Application Notes (AW000649xx000)
# provide more information about this topic.
# The bandwidth used by a FireWire camera device can be limited by adjusting the packet size.
maxCamerasToUse = 2
lines = 1
line_buffer = []
line_buffer2 = []
# The exit code of the sample application.
exitCode = 0
count = 0
img2 = []
try:

    # Get the transport layer factory.
    tlFactory = pylon.TlFactory.GetInstance()

    # Get all attached devices and exit application if no device is found.
    devices = tlFactory.EnumerateDevices()
    if len(devices) == 0:
        raise pylon.RuntimeException("No camera present.")

    # Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
    cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))
    # maxWidth = cameras.WidthMax.Value
    # cameras.Width.Value = maxWidth
    # cameras.Height.Value(500)
    l = cameras.GetSize()

    # Create and attach all Pylon Devices.
    for i, cam in enumerate(cameras):
        print(i)
        cam.Attach(tlFactory.CreateDevice(devices[i]))
        # Print the model name of the camera.
        print("Using device ", cam.GetDeviceInfo().GetModelName())

    # Starts grabbing for all cameras starting with index 0. The grabbing
    # is started for one camera after the other. That's why the images of all
    # cameras are not taken at the same time.
    # However, a hardware trigger setup can be used to cause all cameras to grab images synchronously.
    # According to their default configuration, the cameras are
    # set up for free-running continuous acquisition.
    # print(cameras.WidthMax.Value)
    cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    while True:
        # Grab c_countOfImagesToGrab from the cameras.
        for i in range(lines):
            if not cameras.IsGrabbing():
                break

            grabResult = cameras.RetrieveResult(10000, pylon.TimeoutHandling_ThrowException)
            
            # When the cameras in the array are created the camera context value
            # is set to the index of the camera in the array.
            # The camera context is a user settable value.
            # This value is attached to each grab result and can be used
            # to determine the camera that produced the grab result.
            cameraContextValue = grabResult.GetCameraContext()

            # Print the index and the model name of the camera.
            print("Camera ", cameraContextValue, ": ", cameras[cameraContextValue].GetDeviceInfo().GetModelName())

            # Now, the image data can be processed.
            print("GrabSucceeded: ", grabResult.GrabSucceeded())
            print("SizeX: ", grabResult.GetWidth())
            print("SizeY: ", grabResult.GetHeight())
            img = grabResult.GetArray()
            if cameraContextValue == 0:
                line_buffer.append(img)
            else:
                line_buffer2.append(img)
            
            if line_buffer:
                img = np.vstack(line_buffer)
                cv2.imwrite(f'test_{count}.png', img)
                line_buffer = []
                count += 1
            if line_buffer2:
                img2 = np.vstack(line_buffer2)
                count += 1
                cv2.imwrite(f'test2_{count}.png', img)
                line_buffer2 = []
            cv2.imshow('title', img)
            cv2.imshow('title2', img2)
            k = cv2.waitKey(1)
            if k == 27:
                break
            # Release the grab result to free the buffer for grabbing.
            grabResult.Release()

            print("Gray value of first pixel: ", img[0, 0])

except genicam.GenericException as e:
    # Error handling
    print("An exception occurred.", e)
    exitCode = 1
    cameras.StopGrabbing()

# Comment the following two lines to disable waiting on exit.
sys.exit(exitCode)