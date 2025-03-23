from typing import List, Dict, Any, Optional
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain.chains import LLMChain
# from langchain_ollama import ChatOllama
import requests
import sys
import os
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'voice_assistant')))
from langchain_llm import CustomLLM2

# model = CustomLLM2()

@dataclass
class ConservationAnalysis:
    traditional_practice: List[str]
    traditional_efficiency: List[str]
    traditional_description: List[str]
    modern_practice: List[str]
    improved_efficiency: List[str]
    modern_description: List[str]

class WaterConservationAnalyzer:

    def __init__(self):

        self.llm = CustomLLM2()

        self._setup_schemas()
        
        self._setup_prompt_template()

    def _setup_schemas(self) -> None:
        self.response_schemas = [
            ResponseSchema(
                name="traditional_practice",
                description="A list of top 3 traditional water conservation practices"
            ),
            ResponseSchema(
                name="traditional_efficiency",
                description="A list of efficiency ratings for the traditional practices"
            ),
            ResponseSchema(
                name="traditional_description",
                description="A list of descriptions of the traditional water conservation methods"
            ),
            ResponseSchema(
                name="modern_practice",
                description="A list of modern water conservation practices that improve upon the traditional ones"
            ),
            ResponseSchema(
                name="improved_efficiency",
                description="A list of improved efficiency ratings of the modern methods compared to the traditional ones"
            ),
            ResponseSchema(
                name="modern_description",
                description="A list of descriptions of the modern water conservation techniques"
            ),
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)

    def _setup_prompt_template(self) -> None:
        template = """
        The input problem would consist of the conversational history of the user and based on that you have to analyze and compare the top 3 traditional and modern water conservation practices to address cultural resistance.
        
        Guidelines:
        - Identify the top 3 traditional water conservation methods
        - Provide their efficiency ratings and a brief description
        - Compare them to modern techniques that improve upon them
        - Explain the improved efficiency and describe the modern methods
        
        Input problem: {user_input}
        
        {format_instructions}
        """
        
        self.prompt = PromptTemplate(
            input_variables=["user_input"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()},
            template=template
        )

    def get_history(self):
        response = requests.get("http://127.0.0.1:7000/get_conversation_history")
        return response.json()["history"]


    def analyze_practices(self):
        try:
            problem_description = self.get_history()
            formatted_prompt = self.prompt.format(user_input=problem_description)
            
            output = self.llm.invoke(formatted_prompt)
            
            # print(f"THE OUTPUT IS: {output}")
            
            parsed = self.output_parser.parse(output)
            
            return ConservationAnalysis(
                traditional_practice=parsed["traditional_practice"],
                traditional_efficiency=parsed["traditional_efficiency"],
                traditional_description=parsed["traditional_description"],
                modern_practice=parsed["modern_practice"],
                improved_efficiency=parsed["improved_efficiency"],
                modern_description=parsed["modern_description"]
            )
            # return output
        
        except Exception as e:
            print(f"Error processing input: {str(e)}")
            return None
