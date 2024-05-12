import cv2

camera = cv2.VideoCapture(2)
frameWidth = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
frameRate = int(camera.get(cv2.CAP_PROP_FPS))
fourccCode = cv2.VideoWriter_fourcc(*'mp4v')
videoFileName = 'video.mp4'
videoDimensions = (frameWidth, frameHeight)
recordedVideo = cv2.VideoWriter(videoFileName, fourccCode, frameRate, videoDimensions)

while camera.isOpened():
    ret, frame = camera.read()
    if ret:
        frame = cv2.flip(frame, 1)
        recordedVideo.write(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

camera.release()
recordedVideo.release()
cv2.destroyAllWindows()
