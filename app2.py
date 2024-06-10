# Importing Required Packages

import streamlit as st
from streamlit_extras.colored_header import colored_header
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.manager import collect_runs
from dotenv import load_dotenv
import os
import base64
import logging

# Constants
API_VERSION = "2024-04-01-preview"
MODEL_GPT4 = 'gpt-4-aims'
MODEL_GPT4o = 'gpt-4o'
MODEL_GPT35 = 'gpt-35-aims'
TEMPERATURE = 0
MAX_TOKENS_QA = 4000
MAX_TOKENS_RESP = 8000
MAX_TOKENS_4o = 4000
CACHE_CLEAR_MSG = "Cleared app cache"
ERROR_MSG = "Connection aborted. Please generate response template again"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
azure_openai_api_key = os.getenv("OPENAI_API_KEY_AZURE")
azure_endpoint = os.getenv("OPENAI_ENDPOINT_AZURE")
logger.info("Environment variables loaded")

# Set page config and options
st.cache_data.clear()
st.cache_resource.clear()
logger.info(CACHE_CLEAR_MSG)
st.set_page_config(page_title="IT Glue Copilot", page_icon="🤖", layout="wide")

def initialize_session_state():
    if 'clientOrg' not in st.session_state:
        st.session_state['clientOrg'] = ''
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "What you want to know from the IT Glue Copilot?"}]
    if "default_messages" not in st.session_state:
        st.session_state["default_messages"] = [{"role": "assistant", "content": "What you want to know from the IT Glue Copilot?"}]
    logger.info("Initialized session state")

initialize_session_state()

# Initialize components with Azure OpenAI
embeddings = AzureOpenAIEmbeddings(azure_deployment='embeddings-aims', openai_api_version=API_VERSION, azure_endpoint=azure_endpoint, api_key=azure_openai_api_key)

@st.cache_resource
def azure_openai_setup(api_key, endpoint):
    try:
        logger.info("Setting up Azure OpenAI")
        llm_azure_qa = AzureChatOpenAI(model_name=MODEL_GPT4, openai_api_key=api_key, azure_endpoint=endpoint, openai_api_version=API_VERSION, temperature=TEMPERATURE, max_tokens=MAX_TOKENS_QA, model_kwargs={'seed': 123})
        llm_azure_resp = AzureChatOpenAI(model_name=MODEL_GPT35, openai_api_key=api_key, azure_endpoint=endpoint, openai_api_version=API_VERSION, temperature=TEMPERATURE, max_tokens=MAX_TOKENS_RESP, model_kwargs={'seed': 123})
        llm_azure_4o = AzureChatOpenAI(model_name=MODEL_GPT4o, openai_api_key=api_key, azure_endpoint=endpoint, openai_api_version=API_VERSION, temperature=TEMPERATURE, max_tokens=MAX_TOKENS_QA, model_kwargs={'seed': 123})
        logger.info("Azure OpenAI setup completed")
        return llm_azure_qa, llm_azure_resp, llm_azure_4o
    except Exception as e:
        logger.exception("Error setting up Azure OpenAI: %s", e)
        st.error("Failed to set up Azure OpenAI. Please check the logs for details.")

azure_qa, azure_resp, azure_4o = azure_openai_setup(azure_openai_api_key, azure_endpoint)

prompt_template = """
Given the following context, which may include text, images, and tables, provide a detailed and accurate answer to the question. Ensure that your response is based solely on the provided information.

Context:
{context}

Question:
{question}

Please provide a comprehensive and helpful answer.

Answer:
"""

qa_chain = LLMChain(llm=azure_4o, prompt=PromptTemplate.from_template(prompt_template))

# FAISS vector index setup
base_path = os.getcwd()
# mitsui_path = os.path.join(base_path, 'Faiss_Index_IT Glue', 'Index_Mitsui Chemicals America', 'index.faiss')
northpoint_path = os.path.join(base_path, 'Faiss_Index_IT Glue', 'Index_Northpoint Commercial Finance', 'index.faiss')

# mitsui_index = FAISS.load_local(
#     folder_path=os.path.dirname(mitsui_path),
#     index_name='index',
#     embeddings=embeddings,
#     allow_dangerous_deserialization=True
# )

northpoint_index = FAISS.load_local(
    folder_path=os.path.dirname(northpoint_path),
    index_name='index',
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

faiss_indexes = {
    # "Mitsui Chemicals America": mitsui_index,
    "Northpoint Commercial Finance": northpoint_index
}

# Display the main title
with st.sidebar:
    st.image(r"./synoptek.png", width=275)
colored_header(label="IT Glue Copilot 🤖", description="\n", color_name="violet-70")

# Select Client Org name
with st.sidebar:
    client_names = list(faiss_indexes.keys())
    st.session_state['clientOrg'] = st.selectbox("**Select Accounts Name** 🚩", client_names)
    if st.session_state['clientOrg']:
        st.session_state['vector_store'] = faiss_indexes[st.session_state['clientOrg']]
        st.info(f"You are now connected to {st.session_state['clientOrg']} Account!")
    else:
        st.warning("Add client name above")

# Setup memory for app
memory = ConversationBufferMemory(chat_memory=StreamlitChatMessageHistory(key="langchain_messages"), return_messages=True, memory_key="chat_history")

if st.sidebar.button("Clear all messages"):
    st.session_state["messages"] = list(st.session_state["default_messages"])
    st.sidebar.write("All messages have been cleared")

def display_messages():
    if "messages" in st.session_state:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

display_messages()

user_prompt = st.chat_input()

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)
    
    input_dict = {"input": user_prompt}
    try:
        with collect_runs() as cb:
            if st.session_state.messages[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    with st.spinner("Generating answer..."):
                        try:
                            vector_store = st.session_state.get('vector_store')
                            if not vector_store:
                                st.error("Invalid client organization")
                            
                            relevant_docs = vector_store.similarity_search(user_prompt)
                            context = ""
                            relevant_images = []

                            for d in relevant_docs:
                                if d.metadata['type'] == 'text':
                                    context += '[text]' + d.metadata['original_content']
                                elif d.metadata['type'] == 'table':
                                    context += '[table]' + d.metadata['original_content']
                                elif d.metadata['type'] == 'image':
                                    context += '[image]' + d.page_content
                                    relevant_images.append(d.metadata['original_content'])

                            # Include conversation history in the context
                            conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state['messages']])
                            context = conversation_history + "\n\n" + context

                            ai_response = qa_chain.run({'context': context, 'question': user_prompt})
                            st.write(ai_response)
                            logger.info("AI response generated successfully.")

                            if relevant_images:
                                st.subheader("Relevant Images:")
                                cols = st.columns(len(relevant_images))
                                for idx, img in enumerate(relevant_images):
                                    if isinstance(img, str):
                                        try:
                                            img_bytes = base64.b64decode(img)
                                            cols[idx].image(img_bytes, use_column_width=True, width=270)
                                        except Exception as e:
                                            logger.exception("Error decoding and displaying image: %s", e)
                                            st.error("Error decoding and displaying image. Please try again.")
                        except Exception as e:
                            logger.exception("Error during processing: %s", e)
                            st.error(ERROR_MSG)
        new_ai_message = {"role": "assistant", "content": ai_response}
        st.session_state.messages.append(new_ai_message)
        st.session_state.run_id = cb.traced_runs[0].id
        memory.save_context(input_dict, {"output": ai_response})
        logger.info("Session state updated and context saved successfully.")
    except Exception as e:
        logger.exception("Error during the collection of runs or session state update: %s", e)
        st.error(ERROR_MSG)

logger.info("-------------------------------------------------------------------")
