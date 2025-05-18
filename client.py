from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import asyncio

async def main():
    # OLLAMAを使ったLLMの定義
    # ここではローカルで実行中のOLLAMAサーバーを使用
    model = ChatOllama(
        model="qwen3:0.6b", # 使用したいモデル名を指定（llama3、mistral、codellama等）
        temperature=0.7,
        max_tokens=500
    )
    
    # MCPサーバーの定義
    client = await MultiServerMCPClient(
        {
            "stock": {
                "command": "python",
                "args": ["/home/fukuda/Dev/llm_mcp/yahoofinance_server.py"],
                "transport": "stdio",
            }
        }
    ).__aenter__()
    
    # MCPサーバーをツールとして定義
    tools = client.get_tools()
    
    # エージェントの定義(LangGraphでReActエージェントを定義)
    agent = create_react_agent(model, tools)
    
    # 入力プロンプトの定義
    agent_response = await agent.ainvoke({
        "messages": "2025年4月1から2025年4月5日までのOracleの株の日次の終値を取得してください。"
    })
    
    # 出力結果の表示
    print(agent_response)
    
    # エージェントの実行を終了
    await client.__aexit__(None, None, None)

# 非同期関数を実行
if __name__ == "__main__":
    asyncio.run(main())