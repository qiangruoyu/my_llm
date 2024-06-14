from llama_index.llms import ChatMessage, OpenAILike

llm = OpenAILike(
    api_base="http://localhost:1234/v1",
    timeout=600,  # secs
    api_key="loremIpsum",
    is_chat_model=True,
    context_window=32768,
)
chat_history = [
    ChatMessage(role="system", content="You are a bartender."),
    ChatMessage(role="user", content="What do I enjoy drinking?"),
]
output = llm.chat(chat_history)
print(output)
