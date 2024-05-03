import cv2

camera = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'h26')
out = cv2.VideoWriter('new.mp4', fourcc, 20.0, (640, 480))

while camera.isOpened():
    ret, frame = camera.read()
    if ret:
        frame = cv2.flip(frame, 1)
        out.write(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

camera.release()
out.release()
cv2.destroyAllWindows()
