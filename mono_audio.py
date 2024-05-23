from pydub import AudioSegment
import os


def convert_to_mono_and_export(path):
    """
    Convert stereo audio files to mono and export them as WAV files.

    Args:
    - path (str): Path to the directory containing the audio files.
    """
    # Iterate over each file in the directory
    for filename in os.listdir(path):
        if filename.endswith(".wav"):
            file_path = os.path.join(path, filename)
            sound = AudioSegment.from_wav(file_path)

            # Check if the audio is stereo
            if sound.channels > 1:
                # Convert stereo to mono
                sound = sound.set_channels(1)

                # Export the mono audio as WAV
                sound.export(file_path, format="wav")
                print(f"Converted '{filename}' to mono and exported as WAV.")
            else:
                print(f"'{filename}' is already mono, skipping.")
