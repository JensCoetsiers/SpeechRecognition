import os
import re
from scipy.io.wavfile import read as read_wav
import torch
import torchaudio
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


pipe = None


def load_whisper_model():
    global model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"
    # model_id = "openai/whisper-small"
    # model_id = "openai/whisper-medium"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=20,
        torch_dtype=torch_dtype,
        device=device,
    )

    return pipe


def process_audio_transcriptions_with_pipe(data, wav_files, pipe):
    # Convert torch.Tensor objects to NumPy ndarrays
    audio_numpy = [audio_tensor.numpy() for audio_tensor in data]

    # Prepare audio data in batches
    batch_size = 20
    num_batches = (len(data) + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        batch_data = audio_numpy[start_idx:end_idx]
        batch_files = wav_files[start_idx:end_idx]

        print('Working on batch:', batch_files)

        # Call the pipe function to generate text from audio
        result = pipe(batch_data, generate_kwargs={"language": "dutch"})
        # Regular expression pattern to extract text within single quotes
        pattern = re.compile(r"'([^']*)'")

        # Save the transcriptions to the output file after each batch
        with open('./whisper.txt', 'a') as file:
            for idx, translation in enumerate(result):
                filename = batch_files[idx]
                if isinstance(translation, dict):
                    # If translation is a dictionary, extract the 'text' key
                    if 'text' in translation:
                        extracted_text = translation['text'].strip()
                        file.write(f"{filename}|{extracted_text}\n")
                    else:
                        print(
                            f"No 'text' key found in translation dictionary: {translation}")
                else:
                    print(
                        f"Translation is neither a string nor a dictionary: {translation}")
