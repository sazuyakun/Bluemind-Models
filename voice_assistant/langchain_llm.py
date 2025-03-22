from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from typing import Any, List, Optional
from model_classes import LLMModel  # Import your existing LLM
from pydantic import Field

class CustomLLM(LLM):
    """Custom LangChain LLM Wrapper for LLMModel"""

    model: LLMModel = Field(default_factory=LLMModel)  # Define model properly

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = LLMModel()  # Initialize your existing LLM

    @property
    def _llm_type(self) -> str:
        return "custom_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Process input using your LLMModel"""
        response = self.model.get_response(prompt)  # Call your existing model
        return response.strip()
