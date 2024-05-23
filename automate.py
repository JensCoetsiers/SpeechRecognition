import asyncio
import csv
from scipy.io.wavfile import read as read_wav
import torchaudio
import os
import torch
from pydub import AudioSegment


def convert_to_mono_and_export(path):
    """
    Convert stereo audio files to mono and export them as WAV files with 16-bit samples.

    Args:
    - path (str): Path to the directory containing the audio files.
    """
    # Check if path is a directory
    if os.path.isdir(path):
        # Iterate over each file in the directory
        for filename in os.listdir(path):
            if filename.endswith(".wav"):
                file_path = os.path.join(path, filename)
                sound = AudioSegment.from_wav(file_path)

                # Check if the audio is stereo
                if sound.channels > 1:
                    # Convert stereo to mono
                    sound = sound.set_channels(1)

                # Convert to 16-bit samples
                sound = sound.set_sample_width(2)  # 2 bytes = 16 bits

                # Export the audio as WAV
                sound.export(file_path, format="wav")
                # print(f"Converted '{filename}' to mono with 16-bit samples and exported as WAV.")
    else:
        # Assuming it's a CSV file, process accordingly
        with open(path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                file_path = row["wav_filename"]
                if file_path.endswith(".wav"):
                    sound = AudioSegment.from_wav(file_path)

                    # Check if the audio is stereo
                    if sound.channels > 1:
                        # Convert stereo to mono
                        sound = sound.set_channels(1)

                    # Convert to 16-bit samples
                    sound = sound.set_sample_width(2)  # 2 bytes = 16 bits

                    # Export the audio as WAV
                    sound.export(file_path, format="wav")
                    # print(f"Converted '{file_path}' to mono with 16-bit samples and exported as WAV.")
                else:
                    print(f"'{file_path}' is not a WAV file, skipping.")


def process_audio_files(folder_path):
    '''
    This function processes audio files for Seamless and Whisper
    '''
    processed_files = []
    filenames = []
    # Get a list of all WAV files in the directory
    if isinstance(folder_path, list):
        wav_files = folder_path

        for path in wav_files:
            # Check if it's a valid file path
            if os.path.isfile(path) and path.endswith(".wav"):
                processed_audio = process_audio_file(path)
                processed_files.append(processed_audio)
                filenames.append(os.path.basename(path))

    else:
        wav_files = [file for file in os.listdir(
            folder_path) if file.endswith(".wav")]

        for file in wav_files:
            audio_file_path = os.path.join(folder_path, file)
            processed_audio = process_audio_file(audio_file_path)
            processed_files.append(processed_audio)
            filenames.append(file)

    return processed_files, filenames


def load_audio(audio_file_path):
    '''
    THIS IS A HELPER FUNCTION FOR process_audio_file
    loads a single .wav file in.
    '''
    sampling_rate, data = read_wav(audio_file_path)
    return data, sampling_rate


def process_audio_file(audio_file_path):
    '''
    THIS IS A HELPER FUNCTION FOR process_audio_files
    converts a .wav file to a torch tensor and downsample to 16Khz
    '''
    # Load audio data
    data, sampling_rate = load_audio(audio_file_path)

    # Convert audio from numpy array to a torch tensor
    # Convert to floating point type
    audio_tensor = torch.tensor(data, dtype=torch.float32)

    if sampling_rate == 16000:
        return audio_tensor
    elif sampling_rate != 16000:
        # Downsample the audio
        audio_tensor_resampled = torchaudio.functional.resample(
            audio_tensor, orig_freq=sampling_rate, new_freq=16000)
        return audio_tensor_resampled


async def main():
    print("Available models:")
    print("1. Seamless Model")
    print("2. Whisper")
    print("3. Nova 2 (deepgram)")
    print("4. Standard google api")
    print("5. Telephony google api")
    print("6. Chirp google")

    choice = input("Enter your choice (1,2,3,4,5 or 6): ")

    if choice == "1":
        from seamless_model import load_seamless_model, process_audio_transcriptions
        model, processor = load_seamless_model()
    elif choice == "2":
        from whisper_model import process_audio_transcriptions_with_pipe, load_whisper_model
        model = load_whisper_model()
    elif choice == "3":
        from deepgram_model import process_audio
    elif choice == "4":
        from google_model import process_audio
    elif choice == "5":
        from google_model_tel import process_audio
    elif choice == "6":
        from chirp import process_audio

    audio_dir = input("Enter the path to the audio directory: ")
    # Convert stereo audio files to mono if necessary
    convert_to_mono_and_export(audio_dir)
    if audio_dir.endswith(".csv"):
        with open(audio_dir, "r") as file:
            reader = csv.DictReader(file)
            audio_dir = [row["wav_filename"] for row in reader]

    # Process audio files and get translations
    if choice == "1":
        processed_files, filenames = process_audio_files(audio_dir)
        process_audio_transcriptions(
            processed_files, filenames, model, processor)
    elif choice == "2":
        processed_files, filenames = process_audio_files(audio_dir)
        process_audio_transcriptions_with_pipe(
            processed_files, filenames, model)
    elif choice == "3":
        await process_audio(audio_dir)
    elif choice == "4":
        await process_audio(audio_dir)
    elif choice == "5":
        await process_audio(audio_dir)
    elif choice == "6":
        await process_audio(audio_dir)

if __name__ == "__main__":
    asyncio.run(main())
