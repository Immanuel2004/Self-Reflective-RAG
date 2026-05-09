from langchain_community.tools.tavily_search import TavilySearchResults

def web_search_tool(query: str):
    search = TavilySearchResults(k=3)
    return search.run(query)