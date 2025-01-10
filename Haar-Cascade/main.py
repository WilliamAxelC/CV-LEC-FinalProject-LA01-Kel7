# Import library
import os
import cv2

# Initialize label and classifier
class_label = os.listdir('test/person')
person_classifier = cv2.CascadeClassifier('cascade_classifier/person.xml')
hardhat_classifier = cv2.CascadeClassifier('cascade_classifier/hardhat.xml')
vest_classifier = cv2.CascadeClassifier('cascade_classifier/vest.xml')
mask_classifier = cv2.CascadeClassifier('cascade_classifier/mask.xml')

# Set image path and preprocess to gray
image_path = 'test/example/example.webp'
image_bgr = cv2.imread(image_path)
image_gray = cv2.imread(image_path, 0)

# Detect each object
detected_person = person_classifier.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=5)
detected_hardhat = hardhat_classifier.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=5)
detected_vest = vest_classifier.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=5)
detected_mask = mask_classifier.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=5)

# Mark each object
for x,y,w,h in detected_person:
    cv2.rectangle(image_bgr, (x,y), (x+w, y+h), (255,0,0), 1)
    text = 'Person'
    cv2.putText(image_bgr, text, (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
   
for x,y,w,h in detected_hardhat:
    cv2.rectangle(image_bgr, (x,y), (x+w, y+h), (0,255,0), 1)
    text = 'Hard Hat'
    cv2.putText(image_bgr, text, (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1)
   
for x,y,w,h in detected_vest:
    cv2.rectangle(image_bgr, (x,y), (x+w, y+h), (128,0,128), 1)
    text = 'Safety Vest'
    cv2.putText(image_bgr, text, (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (128,0,128), 1)
   
for x,y,w,h in detected_mask:
    cv2.rectangle(image_bgr, (x,y), (x+w, y+h), (128,95,255), 1)
    text = f'Mask'
    cv2.putText(image_bgr, text, (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (128,95,255), 1)

# Show result
cv2.imshow('Safety Detection', image_bgr)
cv2.waitKey(0)

cv2.destroyAllWindows()