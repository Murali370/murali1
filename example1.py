import asyncio
import io
import glob
import os
import sys
import time
from tkinter import font
import uuid
import requests
import json
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

img_file = open('.\images\9.jpeg', 'rb')

response_detection = face_client.face.detect_with_stream(
    image=img_file,
    detection_model='detection_01',
    recognition_model='recognition_04',
    return_face_attributes=['age', 'emotion'],
)


img = Image.open(img_file)
draw = ImageDraw.Draw(img)
for face in response_detection:
    age = face.face_attributes.age
    emotion = face.face_attributes.emotion
    neutral = '{0:.0f}%'.format(emotion.neutral * 100)
    happiness = '{0:.0f}%'.format(emotion.happiness * 100)
    anger = '{0:.0f}%'.format(emotion.anger * 100)
    sadness = '{0:.0f}%'.format(emotion.sadness * 100)


    rect = face.face_rectangle
    left = rect.left
    top = rect.top
    right = rect.width + left
    bottom = rect.height + top
    draw.rectangle(((left, top), (right, bottom)), outline='red', width=6)

    draw.text((right +4, top), 'Age: ' + str(int(age)), fill=(255, 255 , 255))
    draw.text((right +4, top+35), 'Neutral: ' + neutral, fill=(255, 255 , 255))
    draw.text((right +4, top+70), 'Happy: ' + happiness, fill=(255, 255 , 255))
    draw.text((right +4, top+105), 'Sad: ' + sadness, fill=(255, 255 , 255))
    draw.text((right +4, top+140), 'Angry: ' + anger, fill=(255, 255 , 255))


img.show()