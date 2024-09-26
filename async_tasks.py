# fmt: off
import os
import sys
import platform
import select
import asyncio
from httpx import AsyncClient, Timeout
from collections import deque
from colorama import init, Fore, Back, Style
from dotenv import load_dotenv
from datetime import datetime
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
# fmt: on

load_dotenv()
init(autoreset=True)

ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
VOICE_ID = os.getenv('ELEVENLABS_VOICE_ID', '94zOad0g7T7K4oa7zhDq')  # Changed to a default voice ID
#MODEL_ID = 'eleven_turbo_v2'
MODEL_ID = 'eleven_multilingual_v2'

file_increment = 0
audio_queue = deque()

# Define the text_to_speech_queue as a global asyncio Queue that can be accessed by main script
text_to_speech_queue = asyncio.Queue()

shutdown_event = asyncio.Event()

directory = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
os.makedirs(f'output/{directory}', exist_ok=True)


async def get_available_voices():
    url = 'https://api.elevenlabs.io/v1/voices'
    headers = {
        "Accept": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }

    async with AsyncClient() as client:
        response = await client.get(url, headers=headers)
        if response.status_code == 200:
            voices = response.json()['voices']
            print("Available voices:")
            for voice in voices:
                print(f"ID: {voice['voice_id']}, Name: {voice['name']}")
        else:
            print(f"Error getting voices: {response.status_code} - {response.text}")


async def process_text_to_speech(text):
    global file_increment
    filename = f'output/{directory}/{file_increment}.mp3'
    file_increment += 1

    # Check that ELEVENLABS_API_KEY is not None
    if ELEVENLABS_API_KEY is None:
        print(Fore.RED + "Error: ELEVENLABS_API_KEY is not set.")
        return  # Exit the function early or handle the error as appropriate

    # ElevenLabs API details
    url = f'https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}'
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    data = {
        "model_id": MODEL_ID,
        "text": text,
        "voice_settings": {"similarity_boost": 0.8, "stability": 0.5, "style": 0.0, "use_speaker_boost": True}
    }

    timeout = Timeout(30.0, connect=60.0)

    # print(Fore.YELLOW + f"Calling elevenlabs increment: {file_increment}" + Style.RESET_ALL)

    try:
        async with AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=data, headers=headers)

            if response.status_code == 200:
                with open(filename, 'wb') as audio_file:
                    audio_file.write(response.content)
                audio_queue.append(filename)
            else:
                print(f"Error generating speech: {response.status_code} - {response.text}")
                if response.status_code == 400 and "voice_not_found" in response.text:
                    print("The specified voice ID was not found. Here are the available voices:")
                    await get_available_voices()
    except Exception as e:
        print(f"An error occurred while generating speech: {str(e)}")

async def play_audio():
    while True:
        # print ("DEBUG audio_queue:", audio_queue)
        if not pygame.mixer.music.get_busy() and audio_queue:
            pygame.mixer.music.load(audio_queue.popleft())
            pygame.mixer.music.play()
        await asyncio.sleep(0.1)


async def text_to_speech_consumer(text_to_speech_queue):
    while True:
        text = await text_to_speech_queue.get()
        await process_text_to_speech(text)
        text_to_speech_queue.task_done()


class fake_space_event:
    name = "space"

space_event = fake_space_event()


def set_keyboard_handler(kbhan):
    global keyboard_handler
    keyboard_handler = kbhan


async def check_for_kbd_input():
    while True:
        read = select.select([sys.stdin,], [], [], 0)[0]   # don't block forever, allow graceful exit
        if len(read):
            inp = input()
            keyboard_handler(space_event)
        await asyncio.sleep(0.1)


async def start_async_tasks(text_to_speech_queue):
    """Starts asynchronous tasks without directly calling loop.run_forever()."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # No running event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    consumer_task = loop.create_task(
        text_to_speech_consumer(text_to_speech_queue))
    play_task = loop.create_task(play_audio())
    if platform != "win32":
        keyboard_task = loop.create_task(check_for_kbd_input())
    else:
        keyboard_task = None
    return consumer_task, play_task, keyboard_task


async def stop_async_tasks():
    # Cancel all running tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]

    # Gather all tasks to let them finish with cancellation
    await asyncio.gather(*tasks, return_exceptions=True)
