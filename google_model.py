from google.cloud import speech
import os
from glob import glob
import asyncio
import soundfile as sf


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/jensc/AppData/Roaming/gcloud/application_default_credentials.json"

# Adjust wait settings
WAIT_TIME = 10  # Wait for 10 seconds to retry
BATCH_SIZE = 30
RETRY_LIMIT = 3  # Number of times to retry


def transcribe_file(speech_file):
    """Transcribe the given audio file."""
    client = speech.SpeechClient()

    with open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="nl-NL",
        model="latest_long"
    )
    response = client.recognize(config=config, audio=audio)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        return result.alternatives[0].transcript


def sanitize_filename(filename):
    return "".join([c for c in filename if c.isalpha() or c.isdigit() or c in [' ', '_', '-']]).rstrip()


async def process_file(file_path):
    retries = 0
    while retries < RETRY_LIMIT:
        try:
            transcript = transcribe_file(file_path)
            # print(f'Transcript for {file_path}: {transcript}')

            # Extract folder name
            folder_name = os.path.dirname(file_path)

            # Sanitize folder name
            sanitized_folder_name = sanitize_filename(folder_name)

            # Write transcript to file
            with open(f'./google_transcripts_{sanitized_folder_name}.txt', 'a') as f:
                f.write(f"{os.path.basename(file_path)}|{transcript}\n")

            break  # Break out of the loop if successful
        except Exception as e:
            print(
                f'Failed to transcribe {file_path} due to {type(e).__name__}: {e}')
            retries += 1
            if retries < RETRY_LIMIT:
                print(f'Retrying... Attempt {retries}')
                await asyncio.sleep(WAIT_TIME)
            else:
                print(f'Retry limit reached. Moving to next file.')
                break


async def process_batch(batch_files):
    await asyncio.gather(*[process_file(file_path) for file_path in batch_files])


async def process_audio(data):
    """Process audio files asynchronously."""
    if isinstance(data, list):  # If data is a list of file paths
        files = data
    else:
        files = glob(os.path.join(data, '*.wav'))

    # Split files into batches
    num_files = len(files)
    for i in range(0, num_files, BATCH_SIZE):
        batch_files = files[i:i+BATCH_SIZE]
        print(f'Processing batch {i//BATCH_SIZE + 1}...')

        # Process batch asynchronously
        await process_batch(batch_files)
