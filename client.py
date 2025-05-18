from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import asyncio

async def main():
    model = ChatOllama(
        model="qwen3:0.6b",
        temperature=0.7,
        max_tokens=500
    )
    
    client = MultiServerMCPClient(
        {
            "stock": {
                "command": "python",
                "args": ["/home/fukuda/Dev/llm_mcp/yahoofinance_server.py"],
                "transport": "stdio",
            }
        }
    )
    
    # ここでawaitが必要
    tools = await client.get_tools()
    
    agent = create_react_agent(model, tools)
    
    agent_response = await agent.ainvoke({
        "messages": "2025年4月1から2025年4月5日までのOracleの株の日次の終値を取得してください。"
    })
    
    print(agent_response)

if __name__ == "__main__":
    asyncio.run(main())
