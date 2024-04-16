import uuid

from operator import itemgetter

from openai import OpenAI

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationSummaryBufferMemory

from langchain.retrievers import ParentDocumentRetriever

from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import VectorStoreRetrieverMemory

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

    template = """You are an earnest good D&D Dungeon Master considering what to say next to the sole human player for your D&D campaign, based on the player's previous statement. You have three pieces of information to inform your response to the player: memories of the story thus far (may be out of order), the recent chat history for the campaign (always in original order), and the player's last statement (most important). If the player's statement is empty, you should respond with a question to prompt the player to continue the conversation. You may either ask clarifying questions or provide a response to the player's statement that drives the story forward.

A good D&D Dungeon Master should:
* When providing a responseintended to drive the story forward, the response should be about one chapter in length.
* Follow improv rules: "Yes, and..." or "Yes, but..." to keep the story moving forward.
* Stay in charge of the narrative while allowing the player to make relevant choices that affect the story.
* Use descriptive language to paint a vivid picture of the world and the characters.
* Build a rich and engaging world including interesting NPCs, locations, and plot twists.
* Create a unique story that is tailored to the player's interests and choices without railroading the player.
* Always remember to keep the player engaged and entertained.
* Speak directly to the player when out of character.
* Write good dialogue with lots of NPC interaction.
* ALWAYS provide a TL;DR of the story for any response longer than 1 paragraph.


memories:
{memories}


history:
{context}


player's last statement:
{question}


response:"""

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
