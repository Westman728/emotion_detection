# Import relevant packages
import cv2
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
from PIL import Image

# Set up local filepaths
face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier = load_model(r'emomodel.h5')
age_classifier = load_model(r'agemodel2.h5')
gen_classifier = load_model(r'genmodel3.h5')

# Set up labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
gender_labels = ['Male', 'Female']

# Defining a function that preprocesses and predicts a frame
def process_and_predict(file):
    im = Image.open(file)
    width, height = im.size
    if len(im.getbands()) > 1:
        im = im.convert('L')
    if width == height:
        im = im.resize((48, 48), Image.LANCZOS)
    else:
        if width > height:
            left = width / 2 - height / 2
            right = width / 2 + height / 2
            top = 0
            bottom = height
            im = im.crop((left, top, right, bottom))
            im = im.resize((48, 48), Image.LANCZOS)
        else:
            left = 0
            right = width
            top = 0
            bottom = width
            im = im.crop((left, top, right, bottom))
            im = im.resize((48, 48), Image.LANCZOS)

    ar = np.asarray(im)
    ar = ar.astype('float32')
    ar /= 255.0
    ar = ar.reshape(-1, 48, 48, 1)

    age = age_classifier.predict(ar)
    gender = np.round(gen_classifier.predict(ar))

    age = int(age[0][0])
    gender = 'male' if gender == 0 else 'female'

    return age, gender

# Actual camera frame window
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    labels = []

    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    gray_pil_frame = pil_frame.convert('L')
    gray_frame = np.array(gray_pil_frame)
    faces = face_classifier.detectMultiScale(gray_frame)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray_frame[y:y+h, x:x+w]

        if np.sum([roi_gray]) != 0:
            # Emotion prediction
            roi_gray_emotion = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi_gray_emotion = roi_gray_emotion.astype('float') / 255.0
            roi_gray_emotion = img_to_array(roi_gray_emotion)
            roi_gray_emotion = np.expand_dims(roi_gray_emotion, axis=0)

            pred_emotion = classifier.predict(roi_gray_emotion)[0]
            label_index_emotion = pred_emotion.argmax()
            label_emotion = emotion_labels[label_index_emotion]

            # Age/Gender predictions
            roi_pil = Image.fromarray(roi_gray)
            roi_pil.save('temp_roi.png')
            age, gender = process_and_predict('temp_roi.png')

            # Displaying emotion text above face
            emotion_text_position = (x, y - 10)
            font_color_emotion = (0, 255, 0)  # Green color
            cv2.putText(frame, f'Emotion: {label_emotion}', emotion_text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, font_color_emotion, 2)

            # Display age and gender above emotion label text
            age_gender_text_position = (x, y - 35)
            font_color_age_gender = (0, 255, 0)  # Green color
            cv2.putText(frame, f'Age: {age}', age_gender_text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, font_color_age_gender, 2)
            cv2.putText(frame, f'Gender: {gender}', (x, y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, font_color_age_gender, 2)

    cv2.imshow('Emotion and Age-Gender Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
