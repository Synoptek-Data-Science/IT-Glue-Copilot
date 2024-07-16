
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
import pyotp
import qrcode
import io
import uuid
from azure.storage.blob import BlobServiceClient
import pandas as pd
from typing import Dict, List
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

# Set page config
st.set_page_config(page_title="AI Support Assistant", page_icon="🤖", layout="centered")

# Load environment variables
azure_openai_api_key = os.getenv("OPENAI_API_KEY_AZURE")
azure_endpoint = os.getenv("OPENAI_ENDPOINT_AZURE")
load_dotenv()

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

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Initialize session state
def initialize_session_state():
    session_vars = {
        'clientOrg': '',
        'messages': [{"role": "assistant", "content": "What do you want to query from AI Support Assistant?"}],
        'default_messages': [{"role": "assistant", "content": "What do you want to query from AI Support Assistant?"}],
        'previous_clientOrg': '',
        'otp_setup_complete': False,
        'otp_verified': False,
        'show_qr_code': False,
        'name': None,
        'authentication_status': None,
        'username': None,
        'rating': 0,
        'show_feedback_form': False,
        'ai_response': '',
        'user_prompt': '',
        'conversation_history': [],
        'selected_conversation': None,
        'start_new_conversation': False,
        'conversation_started': False,
        'conversation_id': None
    }
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default
    logger.info("Initialized session state")

initialize_session_state()

def save_feedback(feedback_data):
    feedback_container_name = "itgluecopilot"
    feedback_blob_name = "feedback.csv"
    feedback_blob_client = blob_service_client.get_blob_client(container=feedback_container_name, blob=feedback_blob_name)
    fieldnames = ["conversation_id", "username", "account_name", "prompt", "conversation", "rating", "comments"]

    try:
        existing_feedback_data = feedback_blob_client.download_blob().readall().decode('utf-8')
        existing_feedback_df = pd.read_csv(io.StringIO(existing_feedback_data))
    except Exception:
        existing_feedback_df = pd.DataFrame(columns=fieldnames)

    new_feedback_df = pd.DataFrame([feedback_data])
    updated_feedback_df = pd.concat([existing_feedback_df, new_feedback_df], ignore_index=True)

    output = io.StringIO()
    updated_feedback_df.to_csv(output, index=False)
    feedback_blob_client.upload_blob(output.getvalue(), overwrite=True)

def load_feedback():
    feedback_blob_name = "feedback.csv"
    feedback_blob_client = blob_service_client.get_blob_client(container="itgluecopilot", blob=feedback_blob_name)
    try:
        feedback_data = feedback_blob_client.download_blob().readall().decode('utf-8')
        feedback_df = pd.read_csv(io.StringIO(feedback_data))
        return feedback_df
    except Exception:
        return pd.DataFrame(columns=["conversation_id", "username", "account_name", "prompt", "conversation", "rating", "comments"])

def analyze_feedback(feedback_df):
    if feedback_df.empty:
        return "No feedback available."

    avg_rating = feedback_df["rating"].mean()
    common_issues = feedback_df["comments"].value_counts()

    feedback_summary = f"Average Rating: {avg_rating}\nCommon Issues:\n"
    for issue, count in common_issues.items():
        feedback_summary += f"{issue}: {count} times\n"
    
    return feedback_summary

def calculate_similarity_scores(feedback_df, user_prompt):
    stop_words = set(stopwords.words('english'))
    
    def preprocess(text):
        return ' '.join([word for word in word_tokenize(text.lower()) if word.isalpha() and word not in stop_words])
    
    # Preprocess user prompt and feedback entries
    user_prompt_processed = preprocess(user_prompt)
    feedback_df['combined_text'] = feedback_df.apply(lambda row: preprocess(row['conversation']) + ' ' + preprocess(row['comments']), axis=1)
    
    # Create TF-IDF vectors
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([user_prompt_processed] + feedback_df['combined_text'].tolist())
    
    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    feedback_df['similarity'] = cosine_similarities * 100  # Convert to percentage
    
    # Log similarity scores
    for index, row in feedback_df.iterrows():
        logger.info(f"Feedback ID: {row['conversation_id']} - Similarity: {row['similarity']:.2f}%")
    
    return feedback_df

def enhance_response_with_feedback(ai_response, feedback_df, similarity_threshold=51):
    filtered_feedback = feedback_df[feedback_df['similarity'] >= similarity_threshold]
    feedback_comments = filtered_feedback['comments'].tolist()
    
    if feedback_comments:
        enhanced_response = f"{ai_response}\n\nBased on user feedback, considering the following points:\n" + "\n".join(feedback_comments)
    else:
        enhanced_response = ai_response
    
    return enhanced_response

connection_string = os.getenv("BLOB_CONNECTION_STRING")
container_name = "itgluecopilot"
blob_name = "config/config.yaml"

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)
blob_client = container_client.get_blob_client(blob_name)
blob_data = blob_client.download_blob().readall()
config = yaml.load(io.BytesIO(blob_data), Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)

if st.session_state["authentication_status"] is None:
    st.title("AI Support Assistant")

name, authentication_status, username = authenticator.login('Login', 'main')

if st.session_state["authentication_status"]:
    user_data = config['credentials']['usernames'][username]
    otp_secret = user_data.get('otp_secret', "")
    if not otp_secret:
        otp_secret = pyotp.random_base32()
        config['credentials']['usernames'][username]['otp_secret'] = otp_secret
        
        updated_blob_data = yaml.dump(config)
        blob_client.upload_blob(updated_blob_data, overwrite=True)

        st.session_state['otp_setup_complete'] = False
        st.session_state['show_qr_code'] = True
        logger.info("Generated new OTP secret and set show_qr_code to True")
    else:
        st.session_state['otp_setup_complete'] = True

    totp = pyotp.TOTP(otp_secret)

    if not st.session_state['otp_verified']:
        if st.session_state['show_qr_code']:
            logger.info("Displaying QR code for initial OTP setup")
            otp_uri = totp.provisioning_uri(name=user_data['email'], issuer_name="AI Support Assistant")
            qr = qrcode.make(otp_uri)
            qr = qr.resize((200, 200))

            st.image(qr, caption="Scan this QR code with your authenticator app")

        otp_input = st.text_input("Enter the OTP from your authenticator app", type="password")
        verify_button_clicked = st.button("Verify OTP")

        if verify_button_clicked:
            if totp.verify(otp_input):
                st.session_state['otp_verified'] = True
                st.session_state['show_qr_code'] = False
                st.experimental_rerun()
            else:
                st.error("Invalid OTP. Please try again.")
    else:
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

        prompt_template = """
        Given the following context, which may include text, images, and tables, provide a detailed and accurate answer to the question. Base your response primarily on the provided information. Only include additional context from external sources if absolutely necessary, and clearly identify it as such.

        Context:
        {context}

        Question:
        {question}

        Give the information as it is given in the document for the steps/process given in the document in as much detail as possible.

        ### Instruction ###
        1. **Base your response primarily on the provided context.** If you must incorporate any additional information from your general knowledge, clearly indicate it as "External Information" and keep it to a minimum.
        2. **Be as specific as possible in your response.** Provide detailed and precise information directly related to the query, using the context provided.
        3. **Validate all information against the provided context.** If any information cannot be validated, clearly state: "This information cannot be validated against the provided context."
        4. **Clearly separate information derived from the context and external information.** Use headings such as "IT Glue Response" and "External Information" to differentiate them.
        5. If the context is insufficient, explicitly state: "The provided context does not contain enough information to answer the question."
        6. **Quantify the accuracy of the information.** Use tags such as "Fully Accurate", "Partially Accurate", or "Inaccurate" based on the validation against the provided context.
        7. Ensure that the response is comprehensive, concise, and helpful, focusing on the most relevant information from the provided context.

        ### Example ###

        #### IT Glue Response ####
        [Your answer based on the given context]
        - Accuracy: [Fully Accurate / Partially Accurate / Inaccurate]
        - Explanation: [Provide a brief explanation of why the information is categorized as such]

        #### External Information ####
        [Additional information, clearly marked as external]
        - Accuracy: [Fully Accurate / Partially Accurate / Inaccurate]
        - Explanation: [Provide a brief explanation of why the information is categorized as such]

        At the end of your answer, list the documents or sources used for the information. Avoid discussing or showcasing images unless they are directly relevant to answering the query.

        Answer:
        """

        qa_chain = LLMChain(llm=azure_4o, prompt=PromptTemplate.from_template(prompt_template))

        def load_faiss_indexes(account_indexes: Dict[str, List[str]]) -> Dict[str, List[FAISS]]:
            indexes = {}
            for account, paths in account_indexes.items():
                indexes[account] = []
                for path in paths:
                    try:
                        print(f"Loading index from: {path}")  # Debugging statement
                        index = FAISS.load_local(
                            folder_path=path,
                            index_name='index',
                            embeddings=embeddings,
                            allow_dangerous_deserialization=True
                        )
                        indexes[account].append(index)
                    except Exception as e:
                        print(f"Error loading index from {path} for account {account}: {e}")
            return indexes

        account_indexes = {
            "Mitsui Chemicals America": [
                r"./Faiss_Index_IT Glue/Index_Mitsui Chemicals America/index1",
                r"./Faiss_Index_IT Glue/Index_Mitsui Chemicals America/index2_ocr"
            ],
            "Northpoint Commercial Finance": [
                r"./Faiss_Index_IT Glue/Index_Northpoint Commercial Finance/index1",
                r"./Faiss_Index_IT Glue/Index_Northpoint Commercial Finance/index2_ocr"
            ]
        }

        faiss_indexes = load_faiss_indexes(account_indexes)

        def search_across_indexes(vector_stores: List[FAISS], query: str):
            all_results = []
            for store in vector_stores:
                results = store.similarity_search(query)
                all_results.extend(results)
            return all_results

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
                    st.session_state["show_feedback_form"] = False
                    st.info(f"You are now connected to {st.session_state['clientOrg']} Account!")
                else:
                    st.warning("Add client name above")

        memory = ConversationBufferMemory(
            chat_memory=StreamlitChatMessageHistory(key="langchain_messages"),
            return_messages=True,
            memory_key="chat_history"
        )
        
        if st.sidebar.button("New Conversation"):
            st.session_state["conversation_id"] = str(uuid.uuid4())
            st.session_state["messages"] = list(st.session_state["default_messages"])
            st.session_state["show_feedback_form"] = False
            st.session_state["conversation_started"] = False
            st.experimental_rerun()

        def display_messages():
            if "messages" in st.session_state:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])

        display_messages()

        user_prompt = st.chat_input("Type your message here")

        if user_prompt:
            st.session_state["user_prompt"] = user_prompt
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
                                        st.error("No Account Name Selected")
                                    else:
                                        relevant_docs = search_across_indexes(vector_store, user_prompt)

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

                                        feedback_df = load_feedback()
                                        feedback_df = calculate_similarity_scores(feedback_df, user_prompt)
                                        feedback_summary = analyze_feedback(feedback_df)
                                        relevant_feedback_comments = enhance_response_with_feedback("", feedback_df, similarity_threshold=51)

                                        initial_response = qa_chain.run({'context': context, 'question': user_prompt})
                                        st.session_state["ai_response"] = enhance_response_with_feedback(initial_response, feedback_df, similarity_threshold=51)

                                        st.write(st.session_state["ai_response"])
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
                new_ai_message = {"role": "assistant", "content": st.session_state["ai_response"]}
                st.session_state.messages.append(new_ai_message)
                if cb.traced_runs:
                    st.session_state.run_id = cb.traced_runs[0].id
                memory.save_context(input_dict, {"output": st.session_state["ai_response"]})
                logger.info("Session state updated and context saved successfully.")
                st.session_state["show_feedback_form"] = True

                if not st.session_state["conversation_started"]:
                    title = st.session_state["messages"][1]["content"][:30] + '...' if len(st.session_state["messages"][1]["content"]) > 30 else st.session_state["messages"][1]["content"]
                    st.session_state["conversation_history"].insert(0, {
                        "title": title,
                        "messages": st.session_state["messages"],
                        "conversation_id": st.session_state["conversation_id"]
                    })
                    st.session_state["conversation_started"] = True

                    if len(st.session_state["conversation_history"]) > 5:
                        st.session_state["conversation_history"] = st.session_state["conversation_history"][:5]
            except Exception as e:
                logger.exception("Error during the collection of runs or session state update: %s", e)
                st.error(ERROR_MSG)
        
        if st.session_state["show_feedback_form"]:
            st.subheader("Feedback")

            smileys = {
                "😀": 5,
                "🙂": 4,
                "😐": 3,
                "🙁": 2,
                "😞": 1
            }

            cols = st.columns(len(smileys))
            for i, (smiley, rating) in enumerate(smileys.items()):
                if cols[i].button(smiley):
                    st.session_state["rating"] = rating

            st.write(f"Selected Rating: {st.session_state['rating']}")

            comments = st.text_area("Additional comments")
            feedback_submitted = st.button("Submit Feedback")

            if feedback_submitted:
                feedback_data = {
                    "conversation_id": st.session_state["conversation_id"],
                    "username": username,
                    "account_name": st.session_state["clientOrg"],
                    "conversation": st.session_state["messages"],
                    "prompt": st.session_state["user_prompt"],
                    "rating": st.session_state["rating"],
                    "comments": comments
                }
                save_feedback(feedback_data)
                st.session_state["show_feedback_form"] = False
                st.session_state["rating"] = 0
                st.session_state["user_prompt"] = ""
                st.session_state["ai_response"] = ""
                st.success("Feedback submitted successfully")

        st.sidebar.markdown("""<div style="height: 4vh;"></div>""", unsafe_allow_html=True)
        st.sidebar.subheader("Conversation History")
        for i, conversation in enumerate(st.session_state["conversation_history"]):
            if st.sidebar.button(conversation["title"], key=f"conversation_{i}"):
                st.session_state["selected_conversation"] = conversation
                st.session_state["messages"] = conversation["messages"]
                st.session_state["conversation_id"] = conversation["conversation_id"]
                st.session_state["show_feedback_form"] = False
                st.experimental_rerun()

        with st.sidebar:
            st.sidebar.markdown("""<div style="height: 16vh;"></div>""", unsafe_allow_html=True)
            st.sidebar.markdown(f'## Hello, *{st.session_state["name"]}*')
            if st.button("Logout", key="logout_button"):
                authenticator.logout('Logout', 'sidebar')
                st.session_state['otp_verified'] = False
                st.experimental_rerun()

else:
    if st.session_state["authentication_status"] == False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] == None:
        st.warning('Please enter your username and password')

logger.info("-------------------------------------------------------------------")

