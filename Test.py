import numpy as np
import cv2
import pickle

#############################################

frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75  # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
# IMPORT THE TRANNIED MODEL
pickle_in = open("trained_model.p", "rb")  ## rb = READ BYTE
model = pickle.load(pickle_in)


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


def getCalssName(classNo):
    if classNo == 0:
        return 'Hiz Limiti 20 km/s'
    elif classNo == 1:
        return 'Hiz Limiti 30 km/s'
    elif classNo == 2:
        return 'Hiz Limiti 50 km/s'
    elif classNo == 3:
        return 'Hiz Limiti 60 km/s'
    elif classNo == 4:
        return 'Hiz Limiti 70 km/s'
    elif classNo == 5:
        return 'Hiz Limiti 80 km/s'
    elif classNo == 6:
        return 'Hiz Limiti 80 km/s'
    elif classNo == 7:
        return 'Hiz Limiti 100 km/s'
    elif classNo == 8:
        return 'Hiz Limiti 120 km/s'
    elif classNo == 9:
        return 'Sollama Yasagi'
    elif classNo == 10:
        return 'Kamyonlar Icin Sollama Yasagi'
    elif classNo == 11:
        return 'Ana Yol Tali Yol'
    elif classNo == 12:
        return 'Oncelikli Yol'
    elif classNo == 13:
        return 'Yol Ver'
    elif classNo == 14:
        return 'Dur'
    elif classNo == 15:
        return 'Arac Giremez'
    elif classNo == 16:
        return 'Kamyon Giremez'
    elif classNo == 17:
        return 'Girisi Olmayan Yol'
    elif classNo == 18:
        return 'Dikkat'
    elif classNo == 19:
        return 'Sola Tehlikeli Viraj'
    elif classNo == 20:
        return 'Saga Tehlikeli Viraj'
    elif classNo == 21:
        return 'Ard Arda Tehlikeli Viraj'
    elif classNo == 22:
        return 'Kasisli Yol'
    elif classNo == 23:
        return 'Kaygan Yol'
    elif classNo == 24:
        return 'Sagdan Daralan Yol'
    elif classNo == 25:
        return 'Yol Calismasi'
    elif classNo == 26:
        return 'Trafik İsaretleri'
    elif classNo == 27:
        return 'Yaya Gecidi'
    elif classNo == 28:
        return 'Okul Gecidi'
    elif classNo == 29:
        return 'Bisiklet Gecidi'
    elif classNo == 30:
        return 'Buzlanma Tehlikesi'
    elif classNo == 31:
        return 'Vahsi Hayvan Gecebilir'
    elif classNo == 32:
        return 'Sinirlama ve Kisitlamalarin Sonu'
    elif classNo == 33:
        return 'Ilerden Saga Mecburi Yon'
    elif classNo == 34:
        return 'Ilerden Saga Mecburi Yon'
    elif classNo == 35:
        return 'Ileri Mecburi Yon'
    elif classNo == 36:
        return 'Ileri veya Saga Mecburi Yon'
    elif classNo == 37:
        return 'Ileri veya Saga Mecburi Yon'
    elif classNo == 38:
        return 'Sagdan Gidiniz'
    elif classNo == 39:
        return 'Soldan Gidiniz'
    elif classNo == 40:
        return 'Ada Etrafinda Donunuz'
    elif classNo == 41:
        return 'Sollama Yasagi Sonu'
    elif classNo == 42:
        return 'Kamnyonlar Icin Sollama Yasagi Sonu'



while True:

    # READ IMAGE
    success, imgOrignal = cap.read()

    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Islenmis Fotograf", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrignal, "Tabela Adi ve Sinifi: ", (20, 35), font, 0.75, (6, 87, 168), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "Olasilik: ", (20, 75), font, 0.75, (6, 87, 168), 2, cv2.LINE_AA)

    # PREDICT IMAGE
    predictions = model.predict(img)
    classIndex = np.argmax(predictions,axis=1)
    probabilityValue = np.amax(predictions)
    if probabilityValue > threshold:
        # print(getCalssName(classIndex))
        cv2.putText(imgOrignal, str(classIndex) + " " + str(getCalssName(classIndex)), (260, 35), font, 0.75,
                    (102, 205, 170), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (120, 75), font, 0.75, (102, 205, 170), 2,
                    cv2.LINE_AA)
    cv2.imshow("Sonuc", imgOrignal)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break