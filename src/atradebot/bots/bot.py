# Unified LLM client manager and caller functions

import asyncio
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed
from typing import Optional
from atradebot.config import *
from atradebot.bots.lmstudio_bot import call_lmstudio, call_lmstudio_

class LLMClientManager:
    """Unified manager for all LLM providers (OpenAI, Claude, Ollama)"""
    
    def __init__(self):
        self.openai_client: Optional[OpenAI] = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize clients based on available API keys and configuration"""
        # Initialize OpenAI client
        if OPENAI_API_KEY:
            self.openai_client = OpenAI(base_url="https://api.openai.com/v1", api_key=OPENAI_API_KEY)
        
    def get_client(self, provider: str):
        """Get the appropriate client for the specified provider for AI API calls only. Local models are handled separately."""
        if provider == LLMProvider.OPENAI:
            if not self.openai_client:
                raise ValueError("OpenAI client not initialized. Please set OPENAI_API_KEY environment variable.")
            return self.openai_client
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    def is_provider_available(self, provider: str) -> bool:
        """Check if a provider is available for use"""
        try:
            self.get_client(provider)
            return True
        except ValueError:
            return False

# Global LLM client manager
llm_manager = LLMClientManager()

@retry(stop=stop_after_attempt(2), wait=wait_fixed(90))
def call_openai(prompt, enable_web_search=None):
    """Call OpenAI API (works for both OpenAI and Ollama)"""

    model = CONFIG.OPENAI_MODEL

    client = llm_manager.get_client(LLMProvider.OPENAI)
    tools = []
    if enable_web_search:
        tools.append({"type": "web_search"})
    response = client.responses.create(
        model=model,
        tools=tools,
        input=prompt,
        text={"verbosity": "low"}
    )
    return response.output_text


async def call_ai(prompt, web_search=None, provider=None, check_fn=None):
    """
    features: web search (MCP), provider selection, response checking with retry

    """
    provider = CONFIG.LLM_PROVIDER if provider is None else provider

    # cloud models
    if provider == LLMProvider.OPENAI:
        print("Calling OpenAI...")
        answer = call_openai(prompt, enable_web_search=web_search)
    # local models
    elif provider == LLMProvider.LMSTUDIO:
        if web_search:
            print("Calling LMStudio with web search...")
            answer = await call_lmstudio(prompt)
        else:
            print("Calling LMStudio...")
            answer = await call_lmstudio_(prompt)
    elif provider == LLMProvider.DEBUG:
        answer = prompt
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

    if check_fn:
        if not check_fn(answer):
            print("LLM response did not pass check will retry")
            for i in range(1, 3): #retry
                answer = await call_ai(prompt, web_search=web_search, provider=provider, check_fn=None)
                if check_fn(answer):
                    break
    return answer

if __name__ == "__main__":

    # Test the configured LLM provider
    test_prompt = 'What are the best growth stocks in the information technology sector?'
    
    try:
        answer = asyncio.run(call_ai(test_prompt))
        print(f"Using {CONFIG.LLM_PROVIDER} provider:")
        print(answer)
    except Exception as e:
        print(f"Error with {CONFIG.LLM_PROVIDER} provider: {e}")
