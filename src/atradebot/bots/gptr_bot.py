from gpt_researcher import GPTResearcher
import asyncio

# create research news document for stocks
# https://github.com/assafelovic/gpt-researcher
async def call_gpt_research(prompt):
    report_type = "research_report"
    # Initialize the researcher
    researcher = GPTResearcher(query=prompt, report_type=report_type, config_path=None)
    # Conduct research on the given query
    await researcher.conduct_research()
    # Write the report
    report = await researcher.write_report()
    
    return report

if __name__ == "__main__":
    query = "What is the latest news on Microsoft?"
    report = asyncio.run(call_gpt_research(query))
    print(report)