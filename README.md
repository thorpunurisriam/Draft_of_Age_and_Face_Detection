# Draft_of_Age_and_Face_Detection
It is used to detect the face and the Age of the person from the Different Areas.
import cv2

# Load pre-trained model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained model for gender and age detection
gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')

# Function to detect gender and age
def detect_gender_and_age(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        face_img = image[y:y+h, x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        
        # Gender detection
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = 'Male' if gender_preds[0].argmax() == 0 else 'Female'
        
        # Age detection
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_preds[0].argmax()
        
        return gender, age

# Example usage
image = cv2.imread('example_image.jpg')
gender, age = detect_gender_and_age(image)
print(f"Gender: {gender}, Age: {age}")
