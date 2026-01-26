"""
Based: https://github.com/huygiatrng/AlpacaTradingAgent

Advanced Trading Agents Debate Example with MCP Web Search Integration

This example demonstrates the more complex debate mechanism used in
the TradingAgentsGraph, showing how multiple agents collaborate in
investment decision-making, with web search integration via MCP.
"""

from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
import os
from datetime import datetime
from atradebot.bots.bot import call_ai
from atradebot.bots.prompts import SEARCH_PROMPT_OUTPUT
from atradebot.config import *

from langchain_mcp_adapters.client import MultiServerMCPClient  
from langchain.agents import create_agent
import asyncio
import json
from functools import partial

from atradebot.utils.utils import is_json_in_text

MCP_client = MultiServerMCPClient( {"websearch": {
            "transport": "http",  # HTTP-based remote server
            "url": "http://127.0.0.1:8000/mcp",
        }})


# Define the trading debate state
class TradingDebateState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    prompt: str
    analysis_date: str
    
    # Analyst reports (input to debate)
    market_report: str
    sentiment_report: str
    news_report: str
    fundamentals_report: str
    
    # Investment debate state
    bull_history: str
    bear_history: str
    bull_response: str
    bear_response: str
    judge_decision: str
    
    # Risk debate state
    risky_response: str
    safe_response: str
    neutral_response: str
    risk_judge_decision: str
    
    # Final state
    final_trade_decision: str
    debate_rounds: int
    max_rounds: int



class TradingAgentsDebate:
    def __init__(self, max_debate_rounds: int = 3, use_local_llm: bool = False, use_web_search: bool = True):
        # Initialize the LLM - supports both OpenAI and LM Studio
        if use_local_llm:
            # LM Studio local server (default: localhost:1234)
            self.llm = ChatOpenAI(
                model=CONFIG.LMSTUDIO_MODEL,
                temperature=0.3,
                base_url=CONFIG.LMSTUDIO_BASE_URL,
                api_key="not-needed"  # LM Studio doesn't require API key
            )
            print(f"🏠 Using local LM Studio at: {CONFIG.LMSTUDIO_BASE_URL}")
        else:
            # OpenAI API
            self.llm = ChatOpenAI(
                model=CONFIG.OPENAI_MODEL,
                temperature=0.3,
                api_key=OPENAI_API_KEY
            )
            print("🌐 Using OpenAI API")
        
        self.max_debate_rounds = max_debate_rounds
        self.use_web_search = use_web_search
    
    async def _create_trading_graph(self):
        """Create the trading agents graph with multiple debate phases."""

        tools = await MCP_client.get_tools()  
        self.agent_tools = create_agent(model=self.llm,  tools=tools)

    
        workflow = StateGraph(TradingDebateState)
        
        # Analysis phase
        workflow.add_node("market_analyst", self._market_analyst)
        workflow.add_node("sentiment_analyst", self._sentiment_analyst)
        
        # Investment debate phase
        workflow.add_node("bull_researcher", self._bull_researcher)
        workflow.add_node("bear_researcher", self._bear_researcher)
        workflow.add_node("research_manager", self._research_manager)
        workflow.add_node("check_investment_debate", self._check_investment_debate)
        
        # Risk debate phase
        workflow.add_node("portfolio_manager", self._portfolio_manager)
        
        # Define the workflow
        workflow.add_edge(START, "market_analyst")
        workflow.add_edge("market_analyst", "sentiment_analyst")
        workflow.add_edge("sentiment_analyst", "bull_researcher")
        workflow.add_edge("bull_researcher", "bear_researcher")
        workflow.add_edge("bear_researcher", "check_investment_debate")
        
        # Investment debate loop
        workflow.add_conditional_edges(
            "check_investment_debate",
            self._should_continue_investment_debate,
            {
                "continue": "bull_researcher",
                "judge": "research_manager"
            }
        )
        # Move to risk phase
        workflow.add_edge("research_manager", "portfolio_manager")
        workflow.add_edge("portfolio_manager", END)
        self.graph = workflow.compile()
        return self.graph
    
    def _market_analyst(self, state: TradingDebateState) -> TradingDebateState:
        """Analyze market conditions."""
        prompt = f"""
        Today is {state['analysis_date']}. You are a Market Analyst. 
        Use up-to-date market data and trends to answer the question:
        {state['prompt']}.
        
        Provide a brief market analysis covering:
        - Current market trends
        - Technical indicators
        - Price action analysis
        - Market sentiment
        Summarize key insights.
        """
        
        response = asyncio.run(self.agent_tools.ainvoke({"messages": [{"role": "user", "content": prompt}]}))
        response_c = response['messages'][-1].content
        return {
            **state,
            "market_report": response_c,
            "messages": [AIMessage(content=f"Market Analyst: {response_c}", name="market_analyst")]
        }
    
    def _sentiment_analyst(self, state: TradingDebateState) -> TradingDebateState:
        """Analyze social sentiment."""
        prompt = f"""
        Use the web search tool to gather recent social media trends and news sentiment to answer the question:
        {state['prompt']}.
        Use the web search tool to gather recent social media trends and news sentiment. 
        Summarize the overall sentiment.
        """
        

        response = asyncio.run(self.agent_tools.ainvoke({"messages": [{"role": "user", "content": prompt}]}))
        response_c = response['messages'][-1].content
        return {
            **state,
            "sentiment_report": response_c,
            "messages": [AIMessage(content=f"Sentiment Analyst: {response_c}", name="sentiment_analyst")]
        }
    
    def _bull_researcher(self, state: TradingDebateState) -> TradingDebateState:
        """Bull researcher presents bullish arguments."""
        
        # Build context from previous analysis and bear arguments
        context = f"""
        Market Analysis: {state.get('market_report', 'N/A')}
        Sentiment Analysis: {state.get('sentiment_report', 'N/A')}
        """
        
        if state.get('bear_response'):
            context += f"\nBear's Previous Argument: {state['bear_response']}"
        
        prompt = f"""
        You are a Bull Researcher.
        
        {context}
        
        Present a compelling bullish case for investing.
        Focus on:
        - Growth opportunities
        - Positive catalysts
        - Strong fundamentals
        - Why bears are wrong (if applicable)
        
        Be persuasive and back your arguments with reasoning.
        Summarize key insights.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        bull_response = str(response.content)
        
        # Update bull history
        bull_history = state.get('bull_history', '')
        if bull_history:
            bull_history += f"\n\nRound {state.get('debate_rounds', 0) + 1}: {bull_response}"
        else:
            bull_history = f"Round 1: {bull_response}"
        
        return {
            **state,
            "bull_response": bull_response,
            "bull_history": bull_history,
            "messages": [AIMessage(content=f"Bull Researcher: {bull_response}", name="bull_researcher")]
        }
    
    def _bear_researcher(self, state: TradingDebateState) -> TradingDebateState:
        """Bear researcher presents bearish arguments."""
        
        context = f"""
        Market Analysis: {state.get('market_report', 'N/A')}
        Sentiment Analysis: {state.get('sentiment_report', 'N/A')}
        Bull's Argument: {state['bull_response']}
        """
        
        prompt = f"""
        You are a Bear Researcher.
        
        {context}
        
        Present a compelling bearish case against investing.
        Focus on:
        - Risks and challenges
        - Negative catalysts
        - Weak fundamentals
        - Why bulls are wrong
        
        Be persuasive and back your arguments with reasoning.
        Summarize key insights.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        bear_response = str(response.content)
        
        # Update bear history
        bear_history = state.get('bear_history', '')
        if bear_history:
            bear_history += f"\n\nRound {state.get('debate_rounds', 0) + 1}: {bear_response}"
        else:
            bear_history = f"Round 1: {bear_response}"
        
        return {
            **state,
            "bear_response": bear_response,
            "bear_history": bear_history,
            "debate_rounds": state.get('debate_rounds', 0) + 1,
            "messages": [AIMessage(content=f"Bear Researcher: {bear_response}", name="bear_researcher")]
        }
    
    def _check_investment_debate(self, state: TradingDebateState) -> TradingDebateState:
        """Check if investment debate should continue."""
        return state
    
    def _should_continue_investment_debate(self, state: TradingDebateState) -> str:
        """Decide whether to continue the investment debate."""
        if state.get('debate_rounds', 0) >= state.get('max_rounds', self.max_debate_rounds):
            return "judge"
        return "continue"
    
    def _research_manager(self, state: TradingDebateState) -> TradingDebateState:
        """Research manager makes investment decision based on debate."""
        
        prompt = f"""
        You are the Research Manager evaluating the investment debate for the question:
        {state['prompt']}.
                
        Bull Arguments: {state['bull_history']}
        
        Bear Arguments: {state['bear_history']}
        
        Based on the debate between bull and bear researchers:
        1. Evaluate the strength of each side's arguments
        2. Consider market analysis and sentiment
        3. Make an final answer 
        4. Provide reasoning for your decision
        
        Be balanced and objective in your analysis.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return {
            **state,
            "judge_decision": str(response.content),
            "messages": [AIMessage(content=f"Research Manager: {response.content}", name="research_manager")]
        }
    
    def _portfolio_manager(self, state: TradingDebateState) -> TradingDebateState:
        """Final portfolio management decision."""
        prompt = f"""
        You are the Portfolio Manager making the final answer for the question: 
        {state['prompt']}.
        
        Research Recommendation: {state['judge_decision']}
        Make the FINAL answer.
        
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return {
            **state,
            "final_trade_decision": str(response.content),
            "messages": [AIMessage(content=f"Portfolio Manager: {response.content}", name="portfolio_manager")]
        }
    
    def run_agent(self, prompt: str, out_prompt: str) -> dict:
        """Run complete analysis for a ticker."""
        
        initial_state = {
            "prompt": prompt,
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "market_report": "",
            "sentiment_report": "",
            "bull_history": "",
            "bear_history": "",
            "bull_response": "",
            "bear_response": "",
            "judge_decision": "",
            "final_trade_decision": "",
            "debate_rounds": 0,
            "max_rounds": self.max_debate_rounds,
            "messages": []
        }
        
        print(f"🔍 Analyzing {prompt}")
        print("=" * 50)
        
        final_state = None
        for step_output in self.graph.stream(initial_state):
            for node_name, node_output in step_output.items():
                if node_name in ["market_analyst", "sentiment_analyst"]:
                    print(f"\n📊 {node_name.replace('_', ' ').title()}")
                elif node_name in ["bull_researcher", "bear_researcher"]:
                    round_num = node_output.get('debate_rounds', 1)
                    print(f"\n💭 {node_name.replace('_', ' ').title()} - Round {round_num}")
                elif node_name == "research_manager":
                    print(f"\n⚖️  Research Manager Decision")
                elif "analyst" in node_name:
                    print(f"\n🎯 {node_name.replace('_', ' ').title()}")
                elif node_name == "portfolio_manager":
                    print(f"\n🏦 Final Portfolio Decision")
                
                final_state = node_output

        if final_state is None:
            raise ValueError("No final state produced by the trading agents debate graph.")
        
        prompt_ = f"""{final_state['final_trade_decision']}
        {out_prompt}
        """
        print(prompt_)
        answer = asyncio.run(call_ai(prompt_, web_search=False, check_fn=is_json_in_text))

        return answer





def main():
    use_local = True
    use_web_search = True  # Enable web search for sentiment analysis
    
    # Create trading debate system with web search
    trading_system = TradingAgentsDebate(
        max_debate_rounds=2, 
        use_local_llm=use_local,
        use_web_search=use_web_search
    )
    asyncio.run(trading_system._create_trading_graph())

    # Analyze a ticker
    prompt = "Is now a good time to invest in AAPL stock?"
    result = trading_system.run_agent(prompt)
    print(f"\n✅ Analysis complete for {prompt}")
    print(f"📝 Final Decision: {result}...")
    print("\n" + "="*80)
        
if __name__ == "__main__":
    main()