from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor
import torch

processor = None
model = None


def load_seamless_model():
    global processor, model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(
        "facebook/seamless-m4t-v2-large", low_cpu_mem_usage=True)
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        "facebook/seamless-m4t-v2-large")
    model.to(device)

    return model, processor


def process_audio_transcriptions(processed_files, wav_files, model, processor):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for i in range(0, len(processed_files), 20):
        batch_processed_files = processed_files[i:i+20]
        batch_filenames = wav_files[i:i+20]

        print('Working on batch:', batch_filenames)
        audio_inputs = processor(
            audios=batch_processed_files, sampling_rate=16000, return_tensors="pt").to(device)
        output_tokens = model.generate(**audio_inputs, tgt_lang="nld")
        translated_texts = processor.batch_decode(
            output_tokens, skip_special_tokens=True)

        # Save the transcriptions to the output file after each batch
        with open(f'./seamless.txt', 'a') as file:
            for idx, translation in enumerate(translated_texts):
                filename = batch_filenames[idx]
                file.write(f"{filename}|{translation}\n")
