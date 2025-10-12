'''
A simple Program for grabing video from basler camera and converting it to opencv img.
Tested on Basler acA1300-200uc (USB3, linux 64bit , python 3.5)

'''
from pypylon import pylon
import cv2
import numpy as np
# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
camera.TriggerMode.SetValue('Off')
# camera.AcquisitionMode.SetValue('Continuous')
camera.AcquisitionLineRateEnable.SetValue(True)
camera.AcquisitionLineRate.SetValue(2000.0) # max 20000 for acA

converter = pylon.ImageFormatConverter()
# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
buffer = []
count=0
while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        buffer.append(img)
        cv2.namedWindow('title', cv2.WINDOW_NORMAL)
        cv2.imshow('title', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('k'):
            cv2.imwrite(f'test{count}.png', img)
            count += 1
            print(f'saved test{count}.png')
            
        if key == ord('q'):
            break
    grabResult.Release()
img2 = np.vstack(buffer)
cv2.imwrite('test.png', img2)
# Releasing the resource    
camera.StopGrabbing()

cv2.destroyAllWindows()