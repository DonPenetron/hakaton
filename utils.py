import re

speaker_gram_error = [
    "SENAKER", "САНКЕР", "SANAKER", "SONAKER"
]
speaker_gram_error = re.compile("|".join(speaker_gram_error), re.IGNORECASE)

def correct_translation(text: str):
    return speaker_gram_error.sub("СПИКЕР", text)


def extract_speakers(result):
    speakers = set()
    for item in result:
        speaker = item["speaker"]
        if speaker not in speakers:
            speakers.add(speaker)
    speakers = sorted(speakers)
    return speakers