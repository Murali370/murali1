from cmath import rect
import io
import glob
import os
import sys
import time
from tkinter import image_names
import uuid
import json
import requests
from urllib.parse import urlparse
from io import BytesIO
# To install this module, run:
# python -m pip install Pillow
from PIL import Image, ImageDraw, ImageFont
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, QualityForRecognition

credential = json.load(open('credential.json'))
API_KEY = credential['API_KEY']
ENDPOINT = credential['ENDPOINT']

face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(API_KEY))
image_url = 'https://wallpapercave.com/uwp/uwp1736226.jpeg'
image_name = os.path.basename(image_url)

response_detected_faces = face_client.face.detect_with_url(
    image_url,
    detection_model='detection_03' ,
    recognition_model='recognition_04'
)

print(response_detected_faces)

if not response_detected_faces:
    raise Exception('No face detected')
print('Number of people detected: {0}'.format(len(response_detected_faces)))

person1 = response_detected_faces[0]
print(vars(person1))
print(person1.face_rectangle)

response_image = requests.get(image_url)
img = Image.open(io.BytesIO(response_image.content))

draw = ImageDraw.Draw(img)


for face in response_detected_faces:
    rect = face.face_rectangle
    left = rect.left
    top = rect.top
    right = rect.width + left
    bottom = rect.height + top
    draw.rectangle(((left, top), (right, bottom)), outline='red', width=6)

img.show()