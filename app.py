#Imports
import streamlit as st
import yaml
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
from streamlit_extras.colored_header import colored_header
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
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

# Set page config
st.set_page_config(page_title="AI Support Assistant", page_icon="🤖", layout="centered")

# Load config
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)

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

# Clear cache
st.cache_data.clear()
st.cache_resource.clear()
logger.info(CACHE_CLEAR_MSG)

# Initialize session state
def initialize_session_state():
    if 'clientOrg' not in st.session_state:
        st.session_state['clientOrg'] = ''
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "What do you want to query from AI Support Assistant?"}]
    if "default_messages" not in st.session_state:
        st.session_state["default_messages"] = [{"role": "assistant", "content": "What do you want to query from AI Support Assistant?"}]
    if 'previous_clientOrg' not in st.session_state:
        st.session_state['previous_clientOrg'] = ''
    logger.info("Initialized session state")
     
    if "name" not in st.session_state:
        st.session_state.name = None
    
    if "authentication_status" not in st.session_state:
        st.session_state.authentication_status = None
    
    if "username" not in st.session_state:
        st.session_state.username = None

initialize_session_state()

# Show the title when on the login page
if st.session_state["authentication_status"] is None:
    st.title("AI Support Assistant")

# Authentication for App
name, authentication_status, username = authenticator.login('Login', 'main')

if st.session_state["authentication_status"]:
    # Initialize components with Azure OpenAI
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment='embeddings-aims',
        openai_api_version=API_VERSION,
        azure_endpoint=azure_endpoint,
        api_key=azure_openai_api_key
    )

    @st.cache_resource
    def azure_openai_setup(api_key, endpoint):
        try:
            logger.info("Setting up Azure OpenAI")
            llm_azure_qa = AzureChatOpenAI(
                model_name=MODEL_GPT4,
                openai_api_key=api_key,
                azure_endpoint=endpoint,
                openai_api_version=API_VERSION,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS_QA,
                model_kwargs={'seed': 123}
            )
            llm_azure_resp = AzureChatOpenAI(
                model_name=MODEL_GPT35,
                openai_api_key=api_key,
                azure_endpoint=endpoint,
                openai_api_version=API_VERSION,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS_RESP,
                model_kwargs={'seed': 123}
            )
            llm_azure_4o = AzureChatOpenAI(
                model_name=MODEL_GPT4o,
                openai_api_key=api_key,
                azure_endpoint=endpoint,
                openai_api_version=API_VERSION,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS_QA,
                model_kwargs={'seed': 123}
            )
            logger.info("Azure OpenAI setup completed")
            return llm_azure_qa, llm_azure_resp, llm_azure_4o
        except Exception as e:
            logger.exception("Error setting up Azure OpenAI: %s", e)
            st.error("Failed to set up Azure OpenAI. Please check the logs for details.")

    azure_qa, azure_resp, azure_4o = azure_openai_setup(azure_openai_api_key, azure_endpoint)

    # Prompt template for AI responses
    prompt_template = """
    Given the following context, which may include text, images, and tables, provide a detailed and accurate answer to the question. Ensure that your response is based solely on the provided information. 

    Context:
    {context}

    Question:
    {question}

    Please provide a comprehensive and helpful answer.
    Try to give as much information as you can from the provided information, otherwise keep it minimal.

    Answer:
    """

    qa_chain = LLMChain(llm=azure_4o, prompt=PromptTemplate.from_template(prompt_template))

    def load_faiss_indexes():
        return {
            "Mitsui Chemicals America": FAISS.load_local(
                folder_path=r"./Faiss_Index_IT Glue/Index_Mitsui Chemicals America",
                index_name='index',
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            ),
            "Northpoint Commercial Finance": FAISS.load_local(
                folder_path=r"./Faiss_Index_IT Glue/Index_Northpoint Commercial Finance",
                index_name='index',
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
        }

    faiss_indexes = load_faiss_indexes()

    # Display the main title and sidebar content
    with st.sidebar:
        st.image(r"./synoptek.png", width=275)
    colored_header(label="AI Support Assistant 🤖", description="\n", color_name="violet-70")

    with st.sidebar:
        client_names = ["Select an Account Name"] + list(faiss_indexes.keys())
        selected_client = st.selectbox("**Select Account Name** 🚩", client_names)
        if selected_client != st.session_state['previous_clientOrg']:
            st.session_state['clientOrg'] = selected_client
            st.session_state['previous_clientOrg'] = selected_client
            if st.session_state['clientOrg'] and st.session_state['clientOrg'] != "Select an Account Name":
                st.session_state['vector_store'] = faiss_indexes[st.session_state['clientOrg']]
                st.session_state["messages"] = list(st.session_state["default_messages"])
                st.info(f"You are now connected to {st.session_state['clientOrg']} Account!")
            else:
                st.warning("Add client name above")

    # Setup memory for app
    memory = ConversationBufferMemory(
        chat_memory=StreamlitChatMessageHistory(key="langchain_messages"),
        return_messages=True,
        memory_key="chat_history"
    )

    if st.sidebar.button("Clear all messages"):
        st.session_state["messages"] = list(st.session_state["default_messages"])
        st.sidebar.write("All messages have been cleared")

    # Display messages
    def display_messages():
        if "messages" in st.session_state:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

    display_messages()

    # Handle user input
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
    
    # Placeholder for logout button and welcome message at the bottom
    with st.sidebar:
        st.sidebar.markdown("""<div style="height: 18vh;"></div>""", unsafe_allow_html=True)  # Empty space to push content to the bottom
        st.sidebar.markdown(f'## Hello, *{st.session_state["name"]}*')
        authenticator.logout('Logout', 'sidebar')

else:
    # st.title("Welcome to AI Assistant")
    if st.session_state["authentication_status"] == False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] == None:
        st.warning('Please enter your username and password')

logger.info("-------------------------------------------------------------------")