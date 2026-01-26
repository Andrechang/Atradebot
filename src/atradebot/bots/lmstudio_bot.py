# run with websearch MCP server

import asyncio
import logging
from agents import Agent, Runner, trace
from openai import AsyncOpenAI
from agents import Agent, OpenAIResponsesModel, Runner, set_tracing_disabled, function_tool
from agents import Agent, Runner, gen_trace_id, trace
from agents.mcp import MCPServer, MCPServerStreamableHttp
from agents.model_settings import ModelSettings
from atradebot.bots.gptr_bot import call_gpt_research
from atradebot.config import *
from pydantic import BaseModel

set_tracing_disabled(True)
# logging.basicConfig(level=logging.DEBUG)

async def call_lmstudio_websearch(prompt, out_type=None):
    #LMStudio
    gpt_oss_model = OpenAIResponsesModel(
        model=CONFIG.LMSTUDIO_MODEL,
        openai_client=AsyncOpenAI(
            base_url="http://192.168.15.9:1234/v1",
            api_key="lmstudio",
        ),
    ) 

    async with MCPServerStreamableHttp(
        name="Streamable HTTP Python Server",
        params={
            "url": "http://127.0.0.1:8000/mcp",
            "timeout": 60,
        },
        client_session_timeout_seconds=60, # seconds
        cache_tools_list=True,
    ) as server:
        agent = Agent(
            name="Financial Advisor with Web Search",
            instructions="You are an investment advisor equipped with web search tools.",
            mcp_servers=[server],
            model_settings=ModelSettings(tool_choice="auto", temperature=0.3),
            model = gpt_oss_model,
            output_type= out_type,
        )
        # print(f"\n\nRunning: {prompt}")
        result = await Runner.run(starting_agent=agent, input=prompt)
        return result.final_output


async def call_lmstudio_(prompt: str, out_type=None):
    gpt_oss_model = OpenAIResponsesModel(
        model=CONFIG.LMSTUDIO_MODEL,
        openai_client=AsyncOpenAI(
            base_url="http://192.168.15.9:1234/v1",
            api_key="lmstudio",
        ),
    )     
    agent = Agent(
            name="Financial Advisor",
            model = gpt_oss_model,
            output_type= out_type,
            model_settings=ModelSettings(temperature=0.3)
        )
    # print(f"\n\nRunning: {prompt}")
    result = await Runner.run(starting_agent=agent, input=prompt)
    return result.final_output


async def call_lmstudio(prompt: str, out_type=None):
    """
    Async version that handles all calls within a single event loop.
    """
    try:
        return await call_lmstudio_websearch(prompt, out_type=out_type)
    except Exception as e:
        print(f"Warning: MCP websearch failed ({e}), falling back to research + LLM")
        try:
            # fallback - run gpt-research and then LLM  
            prompt_cleaned = prompt.split("Please")[0].strip()
            news = await call_gpt_research(prompt_cleaned)
            combined_prompt = f"{prompt}\n\nRelevant information found:\n{news}"
            return await call_lmstudio_(combined_prompt, out_type=out_type)
        except Exception as fallback_error:
            print(f"Error in fallback: {fallback_error}")
            # Final fallback - just run LLM without search
            return await call_lmstudio_(prompt, out_type=out_type)


if __name__ == "__main__":
    query = "What is the latest news on Microsoft?"
    response = asyncio.run(call_lmstudio(query))
    print("Final response:", response)