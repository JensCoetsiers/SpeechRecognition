import os
import asyncio
from glob import glob
from google.api_core.client_options import ClientOptions
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/jensc/AppData/Roaming/gcloud/application_default_credentials.json"

# Adjust wait settings
WAIT_TIME = 10  # Wait for 10 seconds to retry
BATCH_SIZE = 30
RETRY_LIMIT = 1  # Number of times to retry


def sanitize_filename(filename):
    return "".join([c for c in filename if c.isalpha() or c.isdigit() or c in [' ', '_', '-']]).rstrip()


async def transcribe_chirp_async(audio_file: str):
    """Transcribe an audio file using Chirp asynchronously."""
    retries = 0
    while retries < RETRY_LIMIT:
        try:
            # Instantiates a client
            client = SpeechClient(
                client_options=ClientOptions(
                    api_endpoint="europe-west4-speech.googleapis.com",
                )
            )

            # Reads a file as bytes
            with open(audio_file, "rb") as f:
                content = f.read()

            config = cloud_speech.RecognitionConfig(
                auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
                language_codes=["nl-NL"],
                model="chirp",
            )

            request = cloud_speech.RecognizeRequest(
                recognizer=f"projects/vaulted-botany-416808/locations/europe-west4/recognizers/_",
                config=config,
                content=content,
            )

            # Transcribes the audio into text
            response = client.recognize(request=request)

            transcripts = []
            for result in response.results:
                transcript = result.alternatives[0].transcript
                transcripts.append(transcript)
                # print(f"Transcript: {transcript}")

            # Extract folder name
            folder_name = os.path.dirname(audio_file)

            # Sanitize folder name
            sanitized_folder_name = sanitize_filename(folder_name)

            # Write transcripts to file
            with open(f'./chirp_transcripts_{sanitized_folder_name}.txt', 'a') as f:
                f.write(
                    f"{os.path.basename(audio_file)}|{' |'.join(transcripts)}\n")

            return transcripts
        except Exception as e:
            print(
                f'Failed to transcribe {audio_file} due to {type(e).__name__}: {e}')
            retries += 1
            if retries < RETRY_LIMIT:
                print(f'Retrying... Attempt {retries}')
                await asyncio.sleep(WAIT_TIME)
            else:
                print(f'Retry limit reached. Moving to next file.')
                return None  # Returning None to signify failure


async def process_file(file_path):
    """Process a single file."""
    return await transcribe_chirp_async(file_path)


async def process_batch(batch_files):
    """Process a batch of files asynchronously."""
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
