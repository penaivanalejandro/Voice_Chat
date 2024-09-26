# fmt: off
from summaries import session_one_summary, session_two_summary, session_two_part_two_summary
import os
import asyncio
import keyboard
import platform
import time
import tempfile
import anthropic
import datetime
import sounddevice as sd
import numpy as np
import re
from async_tasks import start_async_tasks, text_to_speech_queue, stop_async_tasks, set_keyboard_handler
from threading import Thread
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from colorama import init, Fore, Style
import subprocess
# Redirect stdout to devnull while importing Pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from async_tasks import get_available_voices
# fmt: on


# Initialization
load_dotenv()
init(autoreset=True)
anthropic_key = os.getenv('ANTHROPIC_KEY')
client = anthropic.Anthropic(api_key=anthropic_key)

# Initialize Pygame for audio playback
pygame.mixer.init()
pygame.init()


# Global variables
conversation_history = []
shutdown_event = asyncio.Event()
model_size = "small"
def get_cuda_version():
    try:
        output = subprocess.check_output(['nvcc', '--version']).decode('utf-8')
        version = re.search(r'release (\S+),', output)
        return version.group(1) if version else "Not found"
    except:
        return "Not found"

def get_cudnn_version():
    try:
        import ctypes
        libcudnn_path = '/usr/local/cuda/targets/x86_64-linux/lib/libcudnn.so'
        if not os.path.exists(libcudnn_path):
            print(f"cuDNN library not found at {libcudnn_path}")
            return "Not found"
        libcudnn = ctypes.CDLL(libcudnn_path)
        version = ctypes.c_int()
        libcudnn.cudnnGetVersion.restype = ctypes.c_int
        libcudnn.cudnnGetVersion(ctypes.byref(version))
        return f"{version.value // 1000}.{(version.value % 1000) // 100}.{version.value % 100}"
    except Exception as e:
        print(f"Error getting cuDNN version: {e}")
        return "Not found"

def detect_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
        else:
            return False, "CUDA is available but no compatible GPU found"
    except ImportError:
        return False, "PyTorch not installed"


def print_system_info():
    cuda_available, cuda_info = detect_cuda()
    cuda_version = get_cuda_version()
    cudnn_version = get_cudnn_version()

    print(Fore.CYAN + "System Information:")
    print(f"CUDA Available: {cuda_available}")
    print(f"CUDA Info: {cuda_info}")
    print(f"CUDA Version: {cuda_version}")
    print(f"cuDNN Version: {cudnn_version}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")

    # Check for specific cuDNN libraries
    cudnn_libs = [
        'libcudnn.so',
        'libcudnn_ops_infer.so.8',
        'libcudnn_ops_train.so.8',
        'libcudnn_cnn_infer.so.8',
        'libcudnn_cnn_train.so.8',
        'libcudnn_adv_infer.so.8',
        'libcudnn_adv_train.so.8'
    ]

    for lib in cudnn_libs:
        path = f"/usr/local/cuda/targets/x86_64-linux/lib/{lib}"
        if os.path.exists(path):
            print(f"{lib} found at {path}")
        else:
            print(f"{lib} not found")

print_system_info()

supports_cuda, cuda_info = detect_cuda()
if supports_cuda:
    compute_type = "float16"
    device = "cuda"
    print(Fore.GREEN + "CUDA is available and will be used for transcription.")
else:
    compute_type = "int8"
    device = "cpu"
    print(Fore.YELLOW + "CUDA is not available. Using CPU for transcription.")

recording_finished = False
is_recording = False
command_key = "<enter>"

class States:
    WAITING_FOR_USER = 1
    RECORDING_USER_INPUT = 2
    PROCESSING_USER_INPUT = 3
    GENERATING_RESPONSE = 4
    PLAYING_RESPONSE = 5


current_state = States.WAITING_FOR_USER

system_message = f'We will continue our conversation. Here is a summary of the first part of our discussion: {session_one_summary} and the second part of our discussion {session_two_summary}...{session_two_part_two_summary}. Please be brief in your replies and focus on the most interesting concepts.'

print("\nClaude's instructions: " + system_message + Fore.YELLOW +
      f"\n\nPress {command_key} to capture your audio and begin the conversation." + Style.RESET_ALL)


def record_audio():
    global is_recording
    fs = 44100  # Sample rate
    duration = 90  # Maximum possible duration, but we can stop earlier
    block_duration = 0.1  # Duration of each audio block in seconds

    # Callback function to be called for each audio block
    def callback(indata, frames, time, status):
        nonlocal audio_data
        if is_recording:
            audio_data.append(indata.copy())

    # Set the flag to True to start recording
    is_recording = True

    audio_data = []
    # Start recording in a non-blocking manner
    with sd.InputStream(callback=callback, samplerate=fs, channels=2, blocksize=int(fs * block_duration)):
        while is_recording and len(audio_data) * block_duration < duration:
            sd.sleep(int(block_duration * 1000))

    # Convert the audio data to a NumPy array
    audio_data = np.concatenate(audio_data, axis=0)

    return audio_data, fs


def on_space_press(event):
    global is_recording, current_state
    if event.name == 'space':
        if current_state == States.WAITING_FOR_USER:
            is_recording = True
            current_state = States.RECORDING_USER_INPUT
            print(Fore.YELLOW + f"Recording started. Press {command_key} to stop.")
        elif current_state == States.RECORDING_USER_INPUT and is_recording:
            is_recording = False  # This will trigger the recording to stop
            print("Recording stopped. Processing input...")
            current_state = States.PROCESSING_USER_INPUT


def transcribe_audio_to_text(audio_data, sample_rate):
    global supports_cuda, compute_type, device
    temp_dir = './input/'
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = tempfile.mktemp(suffix='.wav', dir=temp_dir)
    try:
        write(temp_file_path, sample_rate, audio_data)

        # Use the detected device and compute_type
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

        segments, _ = model.transcribe(temp_file_path)
        transcript = " ".join(segment.text for segment in segments)
        print(Fore.GREEN + "User:", transcript)
        return transcript
    except Exception as e:
        print(Fore.RED + f"Error during transcription: {e}")
        print(Fore.YELLOW + "Detailed error information:")
        import traceback
        traceback.print_exc()
        if supports_cuda:
            print(Fore.YELLOW + "CUDA error encountered. Falling back to CPU.")
            supports_cuda = False
            compute_type = "int8"
            device = "cpu"
            # Retry transcription with CPU
            try:
                model = WhisperModel(model_size, device="cpu", compute_type="int8")
                segments, _ = model.transcribe(temp_file_path)
                transcript = " ".join(segment.text for segment in segments)
                print(Fore.GREEN + "User:", transcript)
                return transcript
            except Exception as e:
                print(Fore.RED + f"Error during CPU transcription: {e}")
                traceback.print_exc()
        return "Error in transcription"
    finally:
        os.remove(temp_file_path)


def generate_and_process_text(user_input, transcription_file):
    # Trim whitespace and check if input is empty
    user_input = user_input.strip()
    if user_input:  # Proceed only if user_input is not empty
        conversation_history.append({"role": "user", "content": user_input})
        with open(transcription_file, "a") as file:
            file.write(f"User: {user_input}\n")
    else:
        print(Fore.RED + "Received empty input, skipping...")
        return  # Skip processing for empty input

    claude_response = ""
    buffer = ""
    min_chunk_size = 150
    splitters = (".", "?", "!")

    # Before streaming, validate conversation_history
    validated_history = [
        msg for msg in conversation_history if msg.get("content").strip()]

    with client.messages.stream(
        max_tokens=1024,
        messages=validated_history,
        system=system_message,
        model="claude-3-opus-20240229",
    ) as stream:
        for text in stream.text_stream:
            print(Fore.CYAN + text + Style.RESET_ALL, end="", flush=True)
            claude_response += text
            buffer += text

            # Check if buffer is ready to be chunked
            last_splitter_pos = max(buffer.rfind(splitter)
                                    for splitter in splitters)
            if len(buffer) >= min_chunk_size and last_splitter_pos != -1:
                chunk = buffer[:last_splitter_pos+1]
                buffer = buffer[last_splitter_pos+1:]

                # Queue this chunk for async TTS processing
                queue_chunk_for_processing(chunk)

    # Process any remaining text in buffer
    if buffer:
        queue_chunk_for_processing(buffer)

    # Append Claude's full response to the conversation history
    conversation_history.append(
        {"role": "assistant", "content": claude_response})

    # Write Claude's completed response to the transcription file
    with open(transcription_file, "a") as file:
        file.write(f"Claude: {claude_response}\n")

    print()  # Newline for separation


def queue_chunk_for_processing(chunk):
    # Remove text within asterisks using regular expression
    chunk = re.sub(r'\*.*?\*', '', chunk)
    # Queue the chunk for asynchronous processing
    text_to_speech_queue.put_nowait(chunk)


def run_async_tasks():
    # Set up the event loop and run the async tasks

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(start_async_tasks(text_to_speech_queue))

    try:
        loop.run_until_complete(shutdown_event.wait())
    finally:
        loop.close()

async def main():
    global current_state, is_recording

    # Check available ElevenLabs voices
    await get_available_voices()

    thread = Thread(target=run_async_tasks)
    thread.start()

    previous_state = None

    transcripts_directory = "transcripts"
    os.makedirs(transcripts_directory, exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    transcription_file = f"{transcripts_directory}/transcription_{timestamp}.txt"
    with open(transcription_file, "w") as file:
        file.write(f"Transcription started at {timestamp}\n\n")

    try:
        if platform == "win32":
            keyboard.on_press(on_space_press)
        else:
            set_keyboard_handler(on_space_press)
        while True:
            if current_state != previous_state:
                previous_state = current_state  # Update previous_state

            if current_state == States.RECORDING_USER_INPUT:
                # Start recording
                recording, fs = record_audio()
                current_state = States.PROCESSING_USER_INPUT

            elif current_state == States.PROCESSING_USER_INPUT:
                # Transcribe and process input
                user_input = transcribe_audio_to_text(recording, fs)
                generate_and_process_text(user_input, transcription_file)
                current_state = States.GENERATING_RESPONSE

            elif current_state == States.GENERATING_RESPONSE:
                if not pygame.mixer.music.get_busy():  # Check if pygame playback is completed
                    current_state = States.WAITING_FOR_USER

            # Add a short sleep to prevent the loop from hogging CPU
            await asyncio.sleep(0.1)

    except KeyboardInterrupt:
        print(Fore.RED + "\nShutting down gracefully...")
        shutdown_event.set()
        thread.join()


if __name__ == "__main__":
    asyncio.run(main())
