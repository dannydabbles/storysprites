import uuid

from operator import itemgetter

from openai import OpenAI

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationSummaryBufferMemory

from langchain.retrievers import ParentDocumentRetriever

from langchain.storage import InMemoryStore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import VectorStoreRetrieverMemory

# Get DALL-E from langchain_community
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper

from chainlit.types import ThreadDict
import chainlit as cl


def setup_runnable():
    chat_memory = cl.user_session.get("chat_memory")  # type: ConversationSummaryBufferMemory
    vector_memory = cl.user_session.get("vector_memory")  # type: VectorStoreRetrieverMemory
    model = ChatOpenAI(
        temperature=0,
        model_name="gpt-4-turbo-preview",
        streaming=True
    )

    template = """You are a D&D Dungeon Master. You should reply to the player's question as if you were a Dungeon Master for this campaign. You have three pieces of information to help you answer the question the context of the campaign (memories), the history of the campaign (history), and the player's question (question). You should use this information to provide a helpful and engaging response. The context of the campaign is presented as a list of memories from past conversations with the player. The history of the campaign is presented as transcript of the recent campaign conversation. The player's question is presented as it was written by the user. You should provide a response to the player's question as a single sentence.

memories:
{memories}

history:
{context}

question:
{question}

answer:"""

    prompt = PromptTemplate.from_template(template)

    runnable = (
        RunnablePassthrough.assign(
            memories=RunnableLambda(vector_memory.load_memory_variables) | itemgetter("history"),
            context=RunnableLambda(chat_memory.load_memory_variables) | itemgetter("history")
        ) |
        prompt |
        model |
        StrOutputParser()
    )
    cl.user_session.set("runnable", runnable)

def get_chat_memory():
    memory = ConversationSummaryBufferMemory(
        llm=ChatOpenAI(streaming=True),
        max_token_limit=1000,
    )

    memory.clear()

    return memory

def get_vector_memory():
    # This text splitter is used to create the parent documents
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    # This text splitter is used to create the child documents
    # It should create documents smaller than the parent
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        collection_name=uuid.uuid4().hex,
        embedding_function=OpenAIEmbeddings()
    )
    # The storage layer for the parent documents
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    memory = VectorStoreRetrieverMemory(
            retriever=retriever,
    )

    memory.clear()

    return memory

def generate_image(human_message, ai_message):
    client = OpenAI()
    response = client.images.generate(
        model="dall-e-3",
        prompt=f"""Create a realistic and cinematic still image based on the following conversation snippet between a human D&D player and an AI Dungeon Master. Include only the image with no text, no text boxes, no boarders, and no poor framing.

Human:
{human_message}
AI:
{ai_message}""",
        n=1,
        size="1024x1024",
    )

    return response.data[0].url


@cl.password_auth_callback
def auth():
    return cl.User(identifier="test")


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("chat_memory", get_chat_memory())
    cl.user_session.set("vector_memory", get_vector_memory())
    setup_runnable()

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    chat_memory = get_chat_memory()
    vector_memory = get_vector_memory()
    root_messages = [m for m in thread["steps"] if m["parentId"] == None]

    message_pair = []
    for message in root_messages:
        message_prefix = None
        if message["type"] == "user_message":
            chat_memory.chat_memory.add_user_message(message["output"])
            message_prefix = "Human"
        else:
            chat_memory.chat_memory.add_ai_message(message["output"])
            message_prefix = "AI"
        message_pair += [(message_prefix, message["output"])]

        if len(message_pair) == 2:
            message_in = message_pair[0]
            message_out = message_pair[1]
            vector_memory.save_context({message_in[0]: message_in[1]}, {message_out[0]: message_out[1]})
            message_pair = message_pair[1:]

    cl.user_session.set("chat_memory", chat_memory)
    cl.user_session.set("vector_memory", vector_memory)
    setup_runnable()


@cl.on_message
async def on_message(message: cl.Message):
    chat_memory = cl.user_session.get("chat_memory")  # type: ConversationSummaryBufferMemory
    vector_memory = cl.user_session.get("vector_memory")

    runnable = cl.user_session.get("runnable")  # type: Runnable

    res = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await res.stream_token(chunk)

    await res.send()

    image_url = generate_image(message.content, res.content)

    image = cl.Image(url=image_url, display="inline")
    res.elements = [image]
    await res.update()

    chat_memory.chat_memory.add_user_message(message.content)
    chat_memory.chat_memory.add_ai_message(res.content)
    chat_memory.save_context({"input": message.content}, {"output": res.content})
    vector_memory.save_context({"Human": message.content}, {"AI": res.content})
