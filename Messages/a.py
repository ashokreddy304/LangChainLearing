from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.memory import ConversationSummaryMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tiktoken

def count_tokens(messages, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    total = 0
    for msg in messages:
        total += len(enc.encode(msg.content))
    return total

def get_recent_messages(messages, k=4):
    return messages[-k:]


summary_memory = ConversationSummaryMemory(llm=llm)

def update_summary(messages):
    for msg in messages:
        summary_memory.chat_memory.add_message(msg)
    return summary_memory.load_memory_variables({})["history"]

#========================================Semantic Retrieval=================
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_texts([], embeddings)

def add_to_vectorstore(messages):
    texts = [m.content for m in messages]
    vectorstore.add_texts(texts)

def get_relevant_messages(query, k=3):
    docs = vectorstore.similarity_search(query, k=k)
    return [HumanMessage(content=d.page_content) for d in docs]


#=======================Merge + Deduplicate==============================
def merge_messages(summary, recent, relevant):
    merged = []

    if summary:
        merged.append(SystemMessage(content=f"Summary: {summary}"))

    merged.extend(relevant)
    merged.extend(recent)

    # simple deduplication
    seen = set()
    unique = []
    for m in merged:
        if m.content not in seen:
            unique.append(m)
            seen.add(m.content)

    return unique

#==============Token Trimming======================
def trim_to_budget(messages, max_tokens=3000):
    while count_tokens(messages) > max_tokens:
        messages.pop(0)  # remove oldest
    return messages

#===============Final Builder Function===============
def build_memory(messages, query, max_tokens=3000):
    recent = get_recent_messages(messages, k=4)

    older = messages[:-4]
    summary = update_summary(older) if older else ""

    add_to_vectorstore(messages)
    relevant = get_relevant_messages(query, k=3)

    merged = merge_messages(summary, recent, relevant)

    final_history = trim_to_budget(merged, max_tokens)

    return final_history

from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Follow rules strictly."),
    MessagesPlaceholder("history"),
    ("human", "{question}")
])

history = build_memory(all_messages, user_query)

response = prompt.invoke({
    "history": history,
    "question": user_query
})