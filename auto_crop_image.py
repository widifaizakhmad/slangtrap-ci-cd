import os
import cv2
import mediapipe as mp

x_ = []
y_ = []

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './autocrop'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 1
dataset_size = 100



cap = cv2.VideoCapture(0)
alphabet_list = [chr(i) for i in range(ord('A'), ord('Z') + 1)]



for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(alphabet_list[j]))):
        os.makedirs(os.path.join(DATA_DIR, str(alphabet_list[j])))

    print('Collecting data for class {}'.format(alphabet_list[j]))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    H, W, _ = frame.shape
                    x_.append(x)
                    y_.append(y)
                
                x1 =int(min(x_) * W) -100
                y1 =int(min(y_) * H) -100
                x2 =int(max(x_) * W) + 100
                y2 =int(max(y_) * H) + 100
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                hand_crop = frame[y1:y2, x1:x2]
                cv2.imshow('frame', frame)
                cv2.waitKey(25)
                cv2.imwrite(os.path.join(DATA_DIR, str(alphabet_list[j]), '{}.jpg'.format(counter)), hand_crop)

        counter += 1

cap.release()
cv2.destroyAllWindows()