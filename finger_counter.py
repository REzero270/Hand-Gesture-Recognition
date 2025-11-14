import cv2
import mediapipe as mp

# --- راه‌اندازی اولیه ---
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# IDهای نوک انگشتان
# (Thumb, Index, Middle, Ring, Pinky)
tip_ids = [4, 8, 12, 16, 20]

# باز کردن وب‌کم
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("خطا: وب‌کم باز نشد.")
    exit()

print("وب‌کم باز شد. 'q' را برای خروج فشار دهید.")

# --- راه‌اندازی مدل تشخیص دست ---
with mp_hands.Hands(
    max_num_hands=1, # فقط یک دست
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # چرخاندن و تبدیل رنگ
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # پردازش
        results = hands.process(image_rgb)

        finger_count = 0
        landmark_list = []
        hand_label = "" # چپ یا راست

        if results.multi_hand_landmarks:
            # --- بخش ۱: پیدا کردن مفاصل و تشخیص چپ/راست ---
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # تشخیص دست چپ یا راست
            try:
                hand_label = results.multi_handedness[0].classification[0].label
            except:
                hand_label = "Unknown"

            # ذخیره مختصات مفاصل
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([id, cx, cy])

            # --- بخش ۲: منطق شمارش ---
            if len(landmark_list) != 0:
                finger_list = [] # 1 برای باز, 0 برای بسته

                # --- منطق ۴ انگشت (اشاره، میانی، انگشتری، کوچک) ---
                # (این منطق برای هر دو دست یکسان است)
                # چک می‌کنیم که نوک انگشت (مثلاً 8) بالاتر از مفصل پایینی (6) باشد
                for id in range(1, 5): # برای انگشتان 1 تا 4
                    if landmark_list[tip_ids[id]][2] < landmark_list[tip_ids[id] - 2][2]:
                        finger_list.append(1) # باز
                    else:
                        finger_list.append(0) # بسته

                # --- منطق انگشت شست (Thumb) ---
                # (این منطق بر اساس چپ یا راست بودن دست تغییر می‌کند)
                if hand_label == "Right":
                    # اگر دست راست بود، نوک شست (4) باید سمت چپ مفصل (3) باشد
                    if landmark_list[tip_ids[0]][1] < landmark_list[tip_ids[0] - 1][1]:
                        finger_list.append(1)
                    else:
                        finger_list.append(0)
                elif hand_label == "Left":
                    # اگر دست چپ بود، نوک شست (4) باید سمت راست مفصل (3) باشد
                    if landmark_list[tip_ids[0]][1] > landmark_list[tip_ids[0] - 1][1]:
                        finger_list.append(1)
                    else:
                        finger_list.append(0)
                
                # شمارش تعداد "1" ها
                finger_count = finger_list.count(1)
            
            # --- بخش ۳: رسم نتایج ---
            # رسم مفاصل
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)
            
            # نمایش اینکه دست چپ است یا راست
            cv2.putText(image, hand_label, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


        # نمایش عدد نهایی
        cv2.rectangle(image, (20, 20), (150, 120), (0, 255, 0), cv2.FILLED)
        cv2.putText(image, str(finger_count), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

        cv2.imshow('Finger Counter - (Press q to quit)', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()