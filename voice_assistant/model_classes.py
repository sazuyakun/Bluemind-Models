from dotenv import load_dotenv
import os
import re

load_dotenv()

os.environ["TEAM_API_KEY"]=os.getenv("AIXPLAIN_ACCESS_KEY")

from aixplain.factories import ModelFactory
import edge_tts
import asyncio

# Automatic speech recognition model
class ASRModel:
    def __init__(self):
        self.model = ModelFactory.get("65eee94812ee0172b4a9a6f7")

    def transcribe(self, audio_path):
        print("ASR Transcribing audio...")
        
        self.result = self.model.run({
            "source_audio": audio_path,
            "language": "en"
        })
        print(f"ASR Transcription completed: {self.result.data}")
        return self.result.data


# Named Entity Recognition model to pass to the vector database: English on Azure-Microsoft
class NERModel:
    def __init__(self):
        self.model = ModelFactory.get("60ddefbc8d38c51c5885f8ba")
    
    def extract_entities(self, text):
        print("NER Extracting Entities...")
        self.result = self.model.run({
            "text": text
        })
        self.entities = []
        for i in self.result.details:
            # print(text[i['boundingBox']['start']: i['boundingBox']['end']])
            # print(i['data'])
            self.entities.append({text[i['boundingBox']['start']: i['boundingBox']['end']]: i['data']})
        
        # Gives output in the format
        # [{'fertilizer': 'Product'},
        # {'irrigation': 'Skill'},
        print(f"NER Extraction completed: {self.entities}")
        return self.entities


# The GOD-LLM: Llama 3.3 70B Versatile on Groq
class LLMModel:
    def __init__(self):
        # self.model = ModelFactory.get("6646261c6eb563165658bbb1")
        self.model = ModelFactory.get("677c16166eb563bb611623c1")

    def get_response(self, text, long_context=False):
        print("LLM generating response")
        if long_context:
            self.result = self.model.run({
                "text": text,
                # "prompt": "<PROMPT_TEXT_DATA>",
                # "context": "<CONTEXT_TEXT_DATA>",
                "max_tokens": "1024",
                # "temperature": "<TEMPERATURE_NUMBER_DATA>",
                # "top_p": "<TOP_P_NUMBER_DATA>",
                # "seed": "<SEED_NUMBER_DATA>",
                # "history": "<HISTORY_TEXT_DATA>"
            })
        else:
            self.result = self.model.run({"text": text, "max_tokens": "64"})
        print("LLM has responded!")
        return self.result.data

    def get_response_for_audio(self, text):
        print("LLM generating response")
        raw_response = self.get_response(text)
        clean_text = raw_response.replace("*", "")
        print("LLM has responded!")
        return raw_response, clean_text.strip()

# For using TTS: Speech Synthesis - English (Australia) - A-FEMALE - Google
class TTSModelAixplain:
    def __init__(self):
        self.model = ModelFactory.get("6171efb6159531495cadf03d")
    
    def speak(self, text):
        print("Conversion to speech...")
        self.result = self.model.run({"text": text})
        print("Conversion to Speech Completed!")
        return self.result.data


# TTS Using Edge_TTS
class TTSModelEdge:
    def __init__(self):
        self.VOICES = [
            # Australian English
            'en-AU-NatashaNeural',  # Female
            'en-AU-WilliamNeural',  # Male

            # Canadian English
            'en-CA-ClaraNeural',  # Female
            'en-CA-LiamNeural',  # Male

            # British English
            'en-GB-LibbyNeural',  # Female
            'en-GB-RyanNeural',   # Male
            'en-GB-SoniaNeural',  # Female (Very natural)
            
            # American English (High-Quality Voices)
            'en-US-JennyNeural',  # Female (Best for natural speech)
            'en-US-GuyNeural',    # Male (Very clear)
            'en-US-AriaNeural',   # Female (Expressive)
            
            # Indian English
            'en-IN-NeerjaNeural',  # Female
            'en-IN-PrabhatNeural'  # Male
        ]
        
        self.VOICE = self.VOICES[10]
        self.OUTPUT_PATH = "output.mp3"
    
    def speak(self, text):
        print("Conversion to speech...")
        self.TEXT = text
        async def amain() -> None:
            communicate = edge_tts.Communicate(self.TEXT, self.VOICE)
            await communicate.save(self.OUTPUT_PATH)

        loop=asyncio.get_event_loop_policy().get_event_loop()
        try:
            loop.run_until_complete(amain())
        finally:
            loop.close()
        print("Conversion to Speech Completed!")