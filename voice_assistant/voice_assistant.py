import threading
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from voice_assistant.model_classes import ASRModel, NERModel, TTSModelEdge, TTSModelAixplain
from voice_assistant.langchain_llm import CustomLLM  # Importing CustomLLM

class VoiceAssistant:
    def __init__(self, max_memory_window: int = 10):
        self.asr_model = None
        self.ner_model = None
        self.llm = None
        self.tts_model = None
        self._initialize_models()
        
        current_date = datetime.now().strftime("%B %d, %Y")

        self.system_prompt = f"""
        "You are a friendly and culturally sensitive chat buddy for farmers, specializing in water conservation technology. Your goal is to assist farmers with practical advice on irrigation techniques, water-saving tools, soil moisture management, and sustainable farming practices while respecting their traditional methods. Use the conversation history to provide personalized suggestions. When introducing modern tech, always relate it to their existing practices, highlight how it preserves cultural values, and provide clear, simple benefits (e.g., 'This tech can work with your stepwell to save 20% more water without changing its traditional role'). Address concerns about cost, complexity, or cultural misalignment by offering relatable examples and reassuring them of the tech's compatibility with their traditions.
        Current date: {current_date}.
        """

        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=max_memory_window
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{query}")
        ])

        def llm_wrapper(query: str, **kwargs) -> str:
            try:
                response = self.llm.invoke(query)
                return response or "Sorry, I couldn't generate a response."
            except:
                return "Sorry, I couldn't process that."

        self.llm_runnable = RunnableLambda(llm_wrapper)

        self.chain = LLMChain(
            llm=self.llm_runnable,
            prompt=self.prompt_template,
            memory=self.memory,
            verbose=True
        )

    def _initialize_models(self) -> None:
        def init_asr(): self.asr_model = ASRModel()
        def init_ner(): self.ner_model = NERModel()
        def init_llm(): self.llm = CustomLLM()
        def init_tts(): self.tts_model = TTSModelAixplain()

        threads = [
            threading.Thread(target=init_asr),
            threading.Thread(target=init_ner),
            threading.Thread(target=init_llm),
            threading.Thread(target=init_tts)
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def forward(self, audio_path):
        audio_to_text = self.asr_model.transcribe(audio_path)

        ner_result = None
        raw_response = None

        def run_ner():
            nonlocal ner_result
            ner_result = self.ner_model.extract_entities(audio_to_text)

        def run_llm():
            nonlocal raw_response
            raw_response = self.chain.invoke({"query": audio_to_text})

        threads = [
            threading.Thread(target=run_ner),
            threading.Thread(target=run_llm)
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        final_response = raw_response['text']
        self.memory.save_context(
            {"query": audio_to_text},
            {"output": final_response or "Sorry, I couldn't process that."}
        )
        

        response_as_audio = self.tts_model.speak(final_response.replace("*", ""))

        return audio_to_text, ner_result, final_response, response_as_audio

    def chat(self, text):
        ner_result = None
        raw_response = None

        def run_ner():
            nonlocal ner_result
            ner_result = self.ner_model.extract_entities(text)

        def run_llm():
            nonlocal raw_response
            raw_response = self.chain.invoke({"query": text})

        threads = [
            threading.Thread(target=run_ner),
            threading.Thread(target=run_llm)
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        final_response = raw_response['text']
        self.memory.save_context(
            {"query": text},
            {"output": final_response or "Sorry, I couldn't process that."}
        )
        return final_response

    def get_conversation_history(self) -> list:
        return self.memory.chat_memory.messages

    def clear_memory(self) -> None:
        self.memory.clear()
    


# assistant = VoiceAssistant()
    
# print("Welcome to the Voice Assistant! Type 'exit' to quit.")
# while True:
#     audio_path = input("Enter the path to your audio file (or type 'exit' to quit): ")
#     if audio_path.lower() == 'exit':
#         break
    
#     text, ner, response, audio_output = assistant.forward(audio_path)
    
#     print(f"Transcription: {text}")
#     print(f"NER Result: {ner}")
#     print(f"Assistant Response: {response}")
#     print(assistant.get_conversation_history()))


# =================================================================

# asr_model = ASRModel()
# result = asr_model.transcribe(audio_path=audio_path)
# # Result: Hello. What is my name?

# ner_model = NERModel()
# text = "Narendra Modi is the prime minister of India."
# result = ner_model.extract_entities(text)
# print(result)
# # Result: [{'Narendra Modi': 'Person'}, {'prime minister': 'PersonType'}, {'India': 'Location'}]

# llm = LLMModel()
# text = "Is Narendra Modi the prime minister of India?"
# result = llm.get_response(text)
# print(result)

# tts_model = TTSModelAixplain()
# audio = tts_model.speak(result)
# print(audio)