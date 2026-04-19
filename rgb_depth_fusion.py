import depthai as dai
import numpy as np
import cv2

# 1. Create pipeline
pipeline = dai.Pipeline()

# 2. Define sources
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
rgb = pipeline.create(dai.node.ColorCamera)
stereo = pipeline.create(dai.node.StereoDepth)

# 3. Define outputs
depthOut = pipeline.create(dai.node.XLinkOut)
rgbOut = pipeline.create(dai.node.XLinkOut)
depthOut.setStreamName("depth")
rgbOut.setStreamName("rgb")

# 4. Configure mono cameras
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# 5. Configure RGB camera
rgb.setPreviewSize(640, 400)
rgb.setInterleaved(False)

# 6. Configure stereo depth
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)  # Align depth to RGB!

# 7. Link nodes
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(depthOut.input)
rgb.preview.link(rgbOut.input)

# 8. Run pipeline
with dai.Device(pipeline) as device:
    depthQueue = device.getOutputQueue("depth", maxSize=4, blocking=False)
    rgbQueue = device.getOutputQueue("rgb", maxSize=4, blocking=False)

    while True:
        depthFrame = depthQueue.get().getFrame()   # uint16, mm
        rgbFrame = rgbQueue.get().getCvFrame()     # uint8, BGR

        # --- Depth processing ---
        # Check minimum depth in center region (corridor wall detection)
        h, w = depthFrame.shape
        cx, cy = w // 2, h // 2
        roi = depthFrame[cy-50:cy+50, cx-50:cx+50]
        min_dist = np.min(roi[roi > 0])  # ignore 0 (invalid pixels)
        print(f"Closest obstacle ahead: {min_dist/1000:.2f} m")

        # --- RGB processing ---
        # (QR detection, red zone detection goes here)

        # --- Visualization ---
        depth_vis = cv2.normalize(depthFrame, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_JET)

        cv2.imshow("RGB", rgbFrame)
        cv2.imshow("Depth", depth_vis)
        if cv2.waitKey(1) == ord('q'):
            break