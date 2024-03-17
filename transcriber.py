import os
import whisperx


class ASRModel:
    def __init__(
            self, device="cuda", compute_type="float16", lang="ru"
    ):
        self.device = device
        self.compute_type = compute_type
        self.lang = lang

        self.model = whisperx.load_model(
            "large-v2", self.device, compute_type=self.compute_type, download_root="/home/greenatom-admin/hakaton/models"
        )
        self.model_a, self.metadata = whisperx.load_align_model(language_code=self.lang, device=self.device)
        self.diarize_model = whisperx.DiarizationPipeline(use_auth_token="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", device=self.device)

    def process(self, audio_filepath: str, batch_size:int=16):
        if not os.path.exists(audio_filepath):
            return {"error": "No such file exists."}
        audio = whisperx.load_audio(audio_filepath)
        print("--- transcribing ---")
        result = self.model.transcribe(audio, batch_size=batch_size, language="ru")
        print("--- aligning ---")
        result = whisperx.align(result["segments"], self.model_a, self.metadata, audio, self.device, return_char_alignments=False)
        print("--- diarizing ---")
        diarize_segments = self.diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        speakers = set()
        for item in result["segments"]:
            speaker = item["speaker"]
            if speaker not in speakers:
                speakers.add(speaker)
        speakers = sorted(speakers)
        result["speakers"] = speakers
        return result