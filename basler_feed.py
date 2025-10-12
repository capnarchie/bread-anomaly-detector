import cv2
from pypylon import pylon
from ultralytics import YOLO

def main():
    # Load your trained YOLO model
    model = YOLO("./yolo/best.pt")
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (640, 640))

    # Create an instant camera object with the camera device found first.
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    # Start grabbing continuously (default strategy: GrabStrategy_OneByOne)
    camera.StartGrabbing(pylon.GrabStrategy_OneByOne)

    # Camera.ImageFormatConverter converts pylon images to OpenCV format
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    cv2.namedWindow('Basler Camera Feed', cv2.WINDOW_NORMAL)

    try:
        while camera.IsGrabbing():
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                # Access the image data and convert to OpenCV format
                image = converter.Convert(grabResult)
                img = image.GetArray()

                # Run YOLO inference on the frame
                results = model(img, imgsz=320, conf=0.5)
                # Draw bounding boxes on the frame
                annotated_frame = results[0].plot()

                cv2.imshow('Basler Camera Feed', cv2.resize(annotated_frame, (640, 640)))
                cv2.imshow('Original Feed', cv2.resize(img, (640, 640)))
                out.write(cv2.resize(img, (640, 640)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            grabResult.Release()
    finally:
        camera.StopGrabbing()
        cv2.destroyAllWindows()
        out.release()

if __name__ == "__main__":
    main()
