import asyncio
import os
from glob import glob
from deepgram import DeepgramClient, PrerecordedOptions

DEEPGRAM_API_KEY = ''


# Adjust wait settings
WAIT_TIME = 5  # Wait for 10 seconds to retry
BATCH_SIZE = 20
RETRY_LIMIT = 3  # Number of times to retry


async def process_file(file_path):
    retries = 0
    while retries < RETRY_LIMIT:
        with open(file_path, 'rb') as buffer_data:
            options = PrerecordedOptions(
                punctuate=True, model="nova-2", language="nl"
            )

            print(f'Requesting transcript for {file_path}...')
            print('Your file may take some time to process.')

            try:
                asyncio.sleep(2)
                response = await asyncio.to_thread(
                    deepgram.listen.prerecorded.v('1').transcribe_file,
                    {'buffer': buffer_data}, options)
                asyncio.sleep(2)
                transcript = response.results.channels[0].alternatives[0].transcript
                print(f'Transcript for {file_path}: {transcript}')

                # Write transcript to file
                with open('./deepgram.txt', 'a') as f:
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
    # Load Deepgram client
    global deepgram
    deepgram = DeepgramClient(DEEPGRAM_API_KEY)

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
