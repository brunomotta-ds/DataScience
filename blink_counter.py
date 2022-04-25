from threading import Thread
import cv2 
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import playsound
from collections import deque

from sqlalchemy import false

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
mp_face_mesh = mp.solutions.face_mesh
index_left_eye = [33, 160, 158, 133, 153, 144]
index_right_eye = [362, 385, 387, 263, 373, 380]
EAR_THRESH = 0.26
#numero de frames para contar como uma qpiscada
NUM_FRAMES = 20
aux_counter = 0
blink_counter = 0
line1 = []
pts_ear = deque(maxlen=64)
i = 0
#alarm = "alarm.wav"
alarm = "buzina.mp3"
alarm_on = False


def sound_alarm(path=alarm):
    # play an alarm sound
    playsound.playsound(alarm)

def drawing_output(frame, coordinates_left_eye, coordinates_right_eye, blink_counter):
    aux_image = np.zeros(frame.shape, np.uint8)
    countrs1 = np.array([coordinates_left_eye])
    countrs2 = np.array([coordinates_right_eye])
    cv2.fillPoly(aux_image, pts=[countrs1], color=(255, 0, 0))
    cv2.fillPoly(aux_image, pts=[countrs2], color=(255, 0, 0))
    output = cv2.addWeighted(frame, 1, aux_image, 0.7, 1)

    cv2.rectangle(output, (0, 0), (200, 50), (255, 0, 0), -1)
    cv2.rectangle(output, (202, 0), (265, 50), (255, 0, 0), 2)
    cv2.putText(output, "Num. Piscadas: ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(output, "{}".format(blink_counter), (220, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 250), 2)
    cv2.imshow("output", output)
    return output

def eye_aspect_ration(coordinates):
    d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5]))
    d_B = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4]))
    d_C = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))

    return (d_A + d_B) / (2 * d_C)

def plotting_ear(pts_ear, line1):
    global figure
    pts = np.linspace(0, 1, 64)
    if line1 == []:
        plt.style.use("ggplot")
        plt.ion()

        figure, ax = plt.subplots()
        line1, = ax.plot(pts, pts_ear)
        plt.ylim(0.1, 0.4)
        plt.xlim(0,1)
        plt.ylabel("EAR", fontsize=18)
    else:
        line1.set_ydata(pts_ear)
        figure.canvas.draw()
        figure.canvas.flush_events()
    return line1



with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1) as face_mesh:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        coordinates_left_eye = []
        coordinates_right_eye = []
        
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                for index in index_left_eye:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordinates_left_eye.append([x, y])
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), 1)
                    cv2.circle(frame, (x, y), 1, (128, 0, 250), 1)
                for index in index_right_eye:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordinates_right_eye.append([x, y])
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), 1)
                    cv2.circle(frame, (x, y), 1, (128, 0, 250), 1)
            ear_left_eye = eye_aspect_ration(coordinates_left_eye)
            ear_right_eye = eye_aspect_ration(coordinates_right_eye)
            ear = (ear_left_eye + ear_right_eye)/2
            print(ear)

            #olhos fechados
            if ear < EAR_THRESH:
                aux_counter += 1

                if aux_counter >= NUM_FRAMES:
                # ligar alarme
                    if not alarm_on:
                        alarm_on = True
                        t = Thread(target=sound_alarm)
                        t.deamon = True
                        t.start()

                    cv2.putText(frame, "[ALERTA] FADIGA!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
                if aux_counter >= NUM_FRAMES:
                    aux_counter = 0
                    alarm_on = False
                    blink_counter +=1
                    print(blink_counter)
            frame = drawing_output(frame, coordinates_left_eye, coordinates_right_eye, blink_counter)
            pts_ear.append(ear)
            if i > 70:
               line1 = plotting_ear(pts_ear, line1)
            i +=1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
