import cv2
import mediapipe as mp

# --- 1. راه‌اندازی اولی ---
print("در حال راه‌اندازی MediaPipe...")
# راه‌اندازی ماژول رسم MediaPipe
mp_drawing = mp.solutions.drawing_utils
# راه‌اندازی ماژول تشخیص دست MediaPipe
mp_hands = mp.solutions.hands

# باز کردن وب‌کم
# اگر وب‌کم دیگری دارید، 0 را به 1 یا 2 تغییر دهید
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("خطا: امکان باز کردن وب‌کم وجود ندارد.")
    exit()

print("وب‌کم با موفقیت باز شد. 'q' را برای خروج فشار دهید.")

# --- 2. راه‌اندازی مدل تشخیص دست ---
# max_num_hands=2 : حداکثر 2 دست را تشخیص بده
# min_detection_confidence=0.7 : حداقل 70% مطمئن باش که این یک دست است
with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as hands:

    # --- 3. حلقه اصلی پردازش فریم ---
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("خطا در خواندن فریم از وب‌کم.")
            continue

        # --- 4. بهینه‌سازی و پردازش ---
        # چرخاندن تصویر (چون وب‌کم معمولاً تصویر آینه‌ای می‌دهد)
        image = cv2.flip(image, 1)
        
        # MediaPipe با فرمت رنگی RGB کار می‌کند، اما OpenCV فرمت BGR دارد
        # پس باید فرمت رنگ را تبدیل کنیم
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # پردازش تصویر برای پیدا کردن دست‌ها
        results = hands.process(image_rgb)

        # --- 5. رسم نتایج ---
        # اگر دستی در تصویر پیدا شد
        if results.multi_hand_landmarks:
            # به ازای هر دستی که پیدا شد
            for hand_landmarks in results.multi_hand_landmarks:
                # مفاصل و خطوط اتصال را روی تصویر رسم کن
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS) # خطوط اتصال مفاصل

        # --- 6. نمایش تصویر ---
        cv2.imshow('MediaPipe Hand Tracker - (Press q to quit)', image)

        # شرط خروج
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# --- 7. آزادسازی منابع ---
cap.release()
cv2.destroyAllWindows()
print("برنامه بسته شد.")