import cv2
import numpy as np
from moildev import Moildev

def load():
    src = 'dataset/a1.avi'
    cap = cv2.VideoCapture(src)
    fps = int(1000 / 20)  # 1000 ms

    alpha = [75, 0, 59, 35]
    beta = [90, 0, -10, 183]
    zoom = 4

    views = []
    for i in range(4):  # loop untuk memasukkan nilai alpha, beta ke moildev, class Moildev tidak bersifat simultan
        moildev = Moildev('calibrasi-entaniya-12.json', 'entaniya_vr220_12')
        map_x, map_y = moildev.maps_anypoint_mode1(alpha[i], beta[i], zoom)     # untuk mendapakan peta tranformasi
        views.append((map_x, map_y))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        transformed_views = [reactangle(frame, *view) for view in views]
        frame = cv2.resize(frame, (500, 400))

        hz1 = np.hstack((transformed_views[0], transformed_views[1]))
        hz2 = np.hstack((transformed_views[2], transformed_views[3]))
        vt = np.vstack((hz1, hz2))

        cv2.imshow('ori', frame)
        cv2.imshow('any_a', vt)

        if cv2.waitKey(fps) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.waitKey(0)

def reactangle(frame, map_x, map_y):   ## metode untuk konvert fisheye ke persegi
    anypoint = cv2.remap(frame, map_x, map_y, cv2.INTER_CUBIC)
    anypoint = cv2.resize(anypoint, (500, 400))

    return anypoint

if __name__ == "__main__":
    load()
