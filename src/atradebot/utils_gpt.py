# https://cookbook.openai.com/examples/gpt_with_vision_for_video_understanding

import cv2  # We're using OpenCV to read video
import base64
import openai
import os
import requests
from openai import OpenAI
from moviepy.editor import *
from PIL import Image
from io import BytesIO

api_key = '<add>'
client = OpenAI(api_key=api_key)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def gen_video_voiceover(video_path, outfile='out.mp4'):
    # generate narration
    video = cv2.VideoCapture(video_path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    print(len(base64Frames), "frames read.")

    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                "These are frames of a video. Create a short voiceover script in the style of David Attenborough. Only include the narration.",
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::10]),
            ],
        },
    ]
    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "api_key": os.environ["OPENAI_API_KEY"],
        "headers": {"Openai-Version": "2020-11-07"},
        "max_tokens": 500,
    }

    result = openai.ChatCompletion.create(**params)
    print(result.choices[0].message.content)

    gen_text_voiceover(result.choices[0].message.content, outfile="tmp.mp3")

#combine video with music
    audioclip = AudioFileClip("tmp.mp3")
    videoclip = VideoFileClip(video_path)
    videoclip.audio = audioclip
    videoclip.write_videofile(outfile)
    videoclip.close()
    audioclip.close()

def gen_text_voiceover(text, outfile='tmp.mp3'):
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )
    response.stream_to_file(outfile)
    
def gen_images(img_path='', mask='', outfile="image.png"):
    # gpt4 not good for generating images
    #for generate thumbnails to add to video: 1280 × 720
    W, H = 1280, 720
    if mask:
        image = Image.open(img_path)
        width, height = 512, 512
        image = image.resize((width, height))
        # Convert the image to a BytesIO object
        byte_stream = BytesIO()
        image.save(byte_stream, format='PNG')
        byte_array = byte_stream.getvalue()
        response = client.images.create_variation(
            image=byte_array,
            mask=open("mask.png", "rb"),
            prompt="add suprised face to image",
            n=1,
            model="dall-e-2",
            size="1024x1024"
        )
    else: 
        response = client.images.generate(
            model="dall-e-3",
            prompt="add suprised face to image with white back ground",
            size="1024x1024",
            quality="standard",
            response_format="b64_json",
            n=1,
        )
    image = response.data[0].b64_json
    img = Image.open(BytesIO(base64.decodebytes(bytes(image, "utf-8"))))
    img.save(outfile)
    return img

def get_img_description(image_path):
    # example to generate image description
    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": "What’s in this image?"
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            ]
        }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    print(response.json())
    return response.json()
    
def get_chat_response(prompt):
    new_msg = [{"role": "system", "content": "You are a helpful assistant."}, 
        {'role': 'user', 'content': prompt}]
    response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=new_msg,
            )
    return response.choices[0].message.content

def get_translation(text, lang):
    if lang != 'en':
        ans = get_chat_response(f'translate into {LANG_OPTIONS[lang]} this text: {text}')
    else:
        ans = text
    return ans

if __name__ == "__main__":
    pass