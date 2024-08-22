# import streamlit as st
# import base64
# import uuid
# import logging
# from typing import List, Dict
# import pandas as pd
# from azure.storage.blob import BlobServiceClient
# from langchain_community.vectorstores import FAISS
# from langchain.memory import ConversationBufferMemory
# from langchain_community.chat_message_histories import StreamlitChatMessageHistory
# from langchain.callbacks.manager import collect_runs
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import io
# import os
# import tempfile

# # Set up logging
# logger = logging.getLogger(__name__)

# # Functions for Chatbot Module

# def initialize_session_state():
#     session_vars = {
#         'clientOrg': '',
#         'messages': [{"role": "assistant", "content": "What do you want to query from AI Support Assistant?"}],
#         'default_messages': [{"role": "assistant", "content": "What do you want to query from AI Support Assistant?"}],
#         'previous_clientOrg': '',
#         'otp_setup_complete': False,
#         'otp_verified': False,
#         'show_qr_code': False,
#         'name': None,
#         'authentication_status': None,
#         'username': None,
#         'rating': 0,
#         'show_feedback_form': False,
#         'ai_response': '',
#         'user_prompt': '',
#         'conversation_history': [],
#         'selected_conversation': None,
#         'start_new_conversation': False,
#         'conversation_started': False,
#         'conversation_id': None,
#         'selected_option': "Home"  
#     }
#     for var, default in session_vars.items():
#         if var not in st.session_state:
#             st.session_state[var] = default
#     logger.info("Initialized session state")

# def save_feedback(feedback_data, blob_service_client):
#     feedback_container_name = "itgluecopilot"
#     feedback_blob_name = "feedback.csv"
#     feedback_blob_client = blob_service_client.get_blob_client(container=feedback_container_name, blob=feedback_blob_name)
#     fieldnames = ["conversation_id", "username", "account_name", "prompt", "conversation", "rating", "comments"]

#     try:
#         existing_feedback_data = feedback_blob_client.download_blob().readall().decode('utf-8')
#         existing_feedback_df = pd.read_csv(io.StringIO(existing_feedback_data))
#     except Exception:
#         existing_feedback_df = pd.DataFrame(columns=fieldnames)

#     new_feedback_df = pd.DataFrame([feedback_data])
#     updated_feedback_df = pd.concat([existing_feedback_df, new_feedback_df], ignore_index=True)

#     output = io.StringIO()
#     updated_feedback_df.to_csv(output, index=False)
#     feedback_blob_client.upload_blob(output.getvalue(), overwrite=True)

# def load_feedback(blob_service_client):
#     feedback_blob_name = "feedback.csv"
#     feedback_blob_client = blob_service_client.get_blob_client(container="itgluecopilot", blob=feedback_blob_name)
#     try:
#         feedback_data = feedback_blob_client.download_blob().readall().decode('utf-8')
#         feedback_df = pd.read_csv(io.StringIO(feedback_data))
#         return feedback_df
#     except Exception:
#         return pd.DataFrame(columns=["conversation_id", "username", "account_name", "prompt", "conversation", "rating", "comments"])

# def analyze_feedback(feedback_df):
#     if feedback_df.empty:
#         return "No feedback available."

#     avg_rating = feedback_df["rating"].mean()
#     common_issues = feedback_df["comments"].value_counts()

#     feedback_summary = f"Average Rating: {avg_rating}\nCommon Issues:\n"
#     for issue, count in common_issues.items():
#         feedback_summary += f"{issue}: {count} times\n"

#     return feedback_summary

# def calculate_similarity_scores(feedback_df, user_prompt):
#     stop_words = set(stopwords.words('english'))

#     def preprocess(text):
#         return ' '.join([word for word in word_tokenize(text.lower()) if word.isalpha() and word not in stop_words])

#     user_prompt_processed = preprocess(user_prompt)
#     feedback_df['combined_text'] = feedback_df.apply(lambda row: preprocess(row['conversation']) + ' ' + preprocess(row['comments']), axis=1)

#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform([user_prompt_processed] + feedback_df['combined_text'].tolist())

#     cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
#     feedback_df['similarity'] = cosine_similarities * 100

#     return feedback_df

# def enhance_response_with_feedback(ai_response, feedback_df, similarity_threshold=48):
#     filtered_feedback = feedback_df[feedback_df['similarity'] >= similarity_threshold]
#     feedback_comments = filtered_feedback['comments'].tolist()

#     if feedback_comments:
#         enhanced_response = f"{ai_response}\n\nBased on user feedback, considering the following points:\n" + "\n".join([f"- {comment}" for comment in feedback_comments])
#     else:
#         enhanced_response = ai_response

#     return enhanced_response

# def load_faiss_indexes(account_indexes: Dict[str, List[str]], embeddings, connection_string: str) -> Dict[str, List[FAISS]]:
#     temp_dirs = {}
#     indexes = {}

#     blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
#     for account, blob_prefixes in account_indexes.items():
#         temp_dirs[account] = []
#         for blob_prefix in blob_prefixes:
#             temp_dir = tempfile.mkdtemp()  # Create a temporary directory
#             container_name, blob_name_prefix = blob_prefix.split("/", 1)
            
#             # Get container client
#             container_client = blob_service_client.get_container_client(container_name)
            
#             # List blobs under the prefix and download them
#             blobs = container_client.list_blobs(name_starts_with=blob_name_prefix)
#             for blob in blobs:
#                 local_file_path = os.path.join(temp_dir, os.path.relpath(blob.name, blob_name_prefix))
#                 os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
#                 blob_client = container_client.get_blob_client(blob)
#                 with open(local_file_path, "wb") as file:
#                     file.write(blob_client.download_blob().readall())
            
#             temp_dirs[account].append(temp_dir)
    
#     # Load FAISS indexes from downloaded directories
#     for account, temp_paths in temp_dirs.items():
#         indexes[account] = []
#         for temp_dir in temp_paths:
#             try:
#                 index = FAISS.load_local(
#                     folder_path=temp_dir,
#                     index_name='index',
#                     embeddings=embeddings,
#                     allow_dangerous_deserialization=True
#                 )
#                 indexes[account].append(index)
#             except Exception as e:
#                 print(f"Error loading index from {temp_dir} for account {account}: {e}")
    
#     return indexes

# def search_across_indexes(vector_stores: List[FAISS], query: str):
#     all_results = []
#     for store in vector_stores:
#         results = store.similarity_search(query, fetch_k=12, k=3)
#         all_results.extend(results)
#     return all_results

# def display_messages():
#     if "messages" in st.session_state:
#         for message in st.session_state.messages:
#             with st.chat_message(message["role"]):
#                 st.write(message["content"])

# # Main chatbot function
# def run_chatbot(faiss_indexes, blob_service_client, embeddings, qa_chain):
#     with st.sidebar:
#         client_names = ["Select an Account Name"] + list(faiss_indexes.keys())
#         selected_client = st.selectbox("**Select Account Name** 🚩", client_names, index=client_names.index(st.session_state.get('clientOrg', "Select an Account Name")))
#         if selected_client != st.session_state['previous_clientOrg']:
#             st.session_state['clientOrg'] = selected_client
#             st.session_state['previous_clientOrg'] = selected_client
#             if st.session_state['clientOrg'] and st.session_state['clientOrg'] != "Select an Account Name":
#                 st.session_state['vector_store'] = faiss_indexes[st.session_state['clientOrg']]
#                 st.session_state["messages"] = list(st.session_state["default_messages"])
#                 st.session_state["show_feedback_form"] = False
#                 st.info(f"You are now connected to {st.session_state['clientOrg']} Account!")
#             else:
#                 st.warning("Add client name above")

#     memory = ConversationBufferMemory(
#         chat_memory=StreamlitChatMessageHistory(key="langchain_messages"),
#         return_messages=True,
#         memory_key="chat_history"
#     )

#     if st.sidebar.button("New Conversation"):
#         st.session_state["conversation_id"] = str(uuid.uuid4())
#         st.session_state["messages"] = list(st.session_state["default_messages"])
#         st.session_state["show_feedback_form"] = False
#         st.session_state["conversation_started"] = False
#         st.experimental_rerun()

#     display_messages()

#     user_prompt = st.chat_input("Type your message here")

#     if user_prompt:
#         st.session_state["user_prompt"] = user_prompt
#         st.session_state.messages.append({"role": "user", "content": user_prompt})
#         with st.chat_message("user"):
#             st.write(user_prompt)

#         input_dict = {"input": user_prompt}
#         try:
#             with collect_runs() as cb:
#                 if st.session_state.messages[-1]["role"] != "assistant":
#                     with st.chat_message("assistant"):
#                         with st.spinner("Generating answer..."):
#                             try:
#                                 vector_store = st.session_state.get('vector_store')
#                                 if not vector_store:
#                                     st.error("No Account Name Selected")
#                                 else:
#                                     relevant_docs = search_across_indexes(vector_store, user_prompt)

#                                     context = ""
#                                     relevant_images = []

#                                     for d in relevant_docs:
#                                         if d.metadata['type'] == 'text':
#                                             context += '[text]' + d.metadata['original_content']
#                                         elif d.metadata['type'] == 'table':
#                                             context += '[table]' + d.metadata['original_content']
#                                         elif d.metadata['type'] == 'image':
#                                             context += '[image]' + d.page_content
#                                             relevant_images.append(d.metadata['original_content'])

#                                     conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state['messages']])
#                                     context = conversation_history + "\n\n" + context

#                                     feedback_df = load_feedback(blob_service_client)
#                                     feedback_df = calculate_similarity_scores(feedback_df, user_prompt)
#                                     feedback_summary = analyze_feedback(feedback_df)
#                                     relevant_feedback_comments = enhance_response_with_feedback("", feedback_df, similarity_threshold=48)

#                                     initial_response = qa_chain.run({'context': context, 'question': user_prompt})
#                                     st.session_state["ai_response"] = enhance_response_with_feedback(initial_response, feedback_df, similarity_threshold=48)

#                                     st.write(st.session_state["ai_response"])
#                                     logger.info("AI response generated successfully.")

#                                     if relevant_images:
#                                         st.subheader("Relevant Images:")
#                                         cols = st.columns(len(relevant_images))
#                                         for idx, img in enumerate(relevant_images):
#                                             if isinstance(img, str):
#                                                 try:
#                                                     img_bytes = base64.b64decode(img)
#                                                     cols[idx].image(img_bytes, use_column_width=True, width=270)
#                                                 except Exception as e:
#                                                     logger.exception("Error decoding and displaying image: %s", e)
#                                                     st.error("Error decoding and displaying image. Please try again.")
#                             except Exception as e:
#                                 logger.exception("Error during processing: %s", e)
#                                 st.error("Connection aborted.")
#             new_ai_message = {"role": "assistant", "content": st.session_state["ai_response"]}
#             st.session_state.messages.append(new_ai_message)
#             if cb.traced_runs:
#                 st.session_state.run_id = cb.traced_runs[0].id
#             memory.save_context(input_dict, {"output": st.session_state["ai_response"]})
#             logger.info("Session state updated and context saved successfully.")
#             st.session_state["show_feedback_form"] = True

#             if not st.session_state["conversation_started"]:
#                 title = st.session_state["messages"][1]["content"][:30] + '...' if len(st.session_state["messages"][1]["content"]) > 30 else st.session_state["messages"][1]["content"]
#                 st.session_state["conversation_history"].insert(0, {
#                     "title": title,
#                     "messages": st.session_state["messages"],
#                     "conversation_id": st.session_state["conversation_id"]
#                 })
#                 st.session_state["conversation_started"] = True

#                 if len(st.session_state["conversation_history"]) > 5:
#                     st.session_state["conversation_history"] = st.session_state["conversation_history"][:5]
#         except Exception as e:
#             logger.exception("Error during the collection of runs or session state update: %s", e)
#             st.error("Connection aborted.")

#     if st.session_state["show_feedback_form"]:
#         st.subheader("Feedback")

#         rating_options = {
#             1: "Inaccurate",
#             2: "Partially Accurate",
#             3: "Accurate"
#         }

#         cols = st.columns(len(rating_options))
#         for idx, (rating, label) in enumerate(rating_options.items()):
#             if cols[idx].button(label):
#                 st.session_state["rating"] = rating

#         st.write(f"Selected Rating: {st.session_state['rating']}")

#         comments = st.text_area("Additional comments")
#         feedback_submitted = st.button("Submit Feedback")

#         if feedback_submitted:
#             if not comments:
#                 comments = "No comments"
#             feedback_data = {
#                 "conversation_id": st.session_state["conversation_id"],
#                 "username": st.session_state["username"],
#                 "account_name": st.session_state["clientOrg"],
#                 "conversation": st.session_state["messages"],
#                 "prompt": st.session_state["user_prompt"],
#                 "rating": st.session_state["rating"],
#                 "comments": comments
#             }
#             save_feedback(feedback_data, blob_service_client)
#             st.session_state["show_feedback_form"] = False
#             st.session_state["rating"] = 0
#             st.session_state["user_prompt"] = ""
#             st.session_state["ai_response"] = ""
#             st.success("Feedback submitted successfully")

#       # Conversation history with trash icon button
#     st.sidebar.markdown("""<div style="height: 1vh;"></div>""", unsafe_allow_html=True)

#     col1, col2 = st.sidebar.columns([4, 2])
#     with col1:
#         st.markdown("""<h3 style='font-size: 18px;'>Conversation History</h3>""", unsafe_allow_html=True)
#     with col2:
#         if st.button("🗑️", key="clear_button"):
#             st.session_state["conversation_history"] = []
#             st.session_state["conversation_deleted"] = True
#             st.experimental_rerun()

#     # Display conversation history buttons
#     for i, conversation in enumerate(st.session_state["conversation_history"]):
#         if st.sidebar.button(conversation["title"], key=f"conversation_{i}"):
#             st.session_state["selected_conversation"] = conversation
#             st.session_state["messages"] = conversation["messages"]
#             st.session_state["conversation_id"] = conversation["conversation_id"]
#             st.session_state["show_feedback_form"] = False
#             st.experimental_rerun()

#     # Show success message if conversations are deleted
#     if "conversation_deleted" in st.session_state and st.session_state["conversation_deleted"]:
#         st.sidebar.success("Conversations deleted!")
#         del st.session_state["conversation_deleted"]


import streamlit as st
import base64
import uuid
import logging
from typing import List, Dict
import pandas as pd
from azure.storage.blob import BlobServiceClient
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.manager import collect_runs
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import os
import tempfile

# Set up logging
logger = logging.getLogger(__name__)

# Functions for Chatbot Module

def initialize_session_state():
    session_vars = {
        'clientOrg': 'Select an Account Name',
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
        'conversation_id': None,
        'selected_option': "Home",
        'faiss_indexes_loaded': False,  # New flag to control FAISS index loading
    }
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default
    logger.info("Initialized session state")

def save_feedback(feedback_data, blob_service_client):
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

def load_feedback(blob_service_client):
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

    user_prompt_processed = preprocess(user_prompt)
    feedback_df['combined_text'] = feedback_df.apply(lambda row: preprocess(row['conversation']) + ' ' + preprocess(row['comments']), axis=1)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([user_prompt_processed] + feedback_df['combined_text'].tolist())

    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    feedback_df['similarity'] = cosine_similarities * 100

    return feedback_df

def enhance_response_with_feedback(ai_response, feedback_df, similarity_threshold=48):
    filtered_feedback = feedback_df[feedback_df['similarity'] >= similarity_threshold]
    feedback_comments = filtered_feedback['comments'].tolist()

    if feedback_comments:
        enhanced_response = f"{ai_response}\n\nBased on user feedback, considering the following points:\n" + "\n".join([f"- {comment}" for comment in feedback_comments])
    else:
        enhanced_response = ai_response

    return enhanced_response

# def load_faiss_indexes(account_indexes: Dict[str, List[str]], embeddings, connection_string: str) -> Dict[str, List[FAISS]]:
#     temp_dirs = {}
#     indexes = {}

#     blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
#     for account, blob_prefixes in account_indexes.items():
#         temp_dirs[account] = []
#         for blob_prefix in blob_prefixes:
#             temp_dir = tempfile.mkdtemp()  # Create a temporary directory
#             container_name, blob_name_prefix = blob_prefix.split("/", 1)
            
#             # Get container client
#             container_client = blob_service_client.get_container_client(container_name)
            
#             # List blobs under the prefix and download them
#             blobs = container_client.list_blobs(name_starts_with=blob_name_prefix)
#             for blob in blobs:
#                 local_file_path = os.path.join(temp_dir, os.path.relpath(blob.name, blob_name_prefix))
#                 os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
#                 blob_client = container_client.get_blob_client(blob)
#                 with open(local_file_path, "wb") as file:
#                     file.write(blob_client.download_blob().readall())
            
#             temp_dirs[account].append(temp_dir)
    
#     # Load FAISS indexes from downloaded directories
#     for account, temp_paths in temp_dirs.items():
#         indexes[account] = []
#         for temp_dir in temp_paths:
#             try:
#                 index = FAISS.load_local(
#                     folder_path=temp_dir,
#                     index_name='index',
#                     embeddings=embeddings,
#                     allow_dangerous_deserialization=True
#                 )
#                 indexes[account].append(index)
#             except Exception as e:
#                 print(f"Error loading index from {temp_dir} for account {account}: {e}")
    
#     return indexes


# import os
# import tempfile
# import shutil  # Import shutil for moving files
# from azure.storage.blob import BlobServiceClient
# from langchain_community.vectorstores import FAISS
# from typing import Dict, List

# def load_faiss_indexes(account_indexes: Dict[str, List[str]], embeddings, connection_string: str) -> Dict[str, List[FAISS]]:
#     indexes = {}
#     local_dir_base = "local_faiss_indexes"  # Base directory for local storage

#     # Create base directory if it doesn't exist
#     if not os.path.exists(local_dir_base):
#         os.makedirs(local_dir_base)

#     for account, blob_prefixes in account_indexes.items():
#         indexes[account] = []
#         for blob_prefix in blob_prefixes:
#             local_dir = os.path.join(local_dir_base, account, os.path.basename(blob_prefix))

#             # Check if the index is already downloaded locally
#             if os.path.exists(local_dir) and os.listdir(local_dir):
#                 # Load FAISS index from local directory
#                 try:
#                     index = FAISS.load_local(
#                         folder_path=local_dir,
#                         index_name='index',
#                         embeddings=embeddings,
#                         allow_dangerous_deserialization=True
#                     )
#                     indexes[account].append(index)
#                 except Exception as e:
#                     print(f"Error loading index from {local_dir} for account {account}: {e}")
#             else:
#                 # If not downloaded, download from Azure Blob Storage
#                 temp_dir = tempfile.mkdtemp()  # Create a temporary directory
#                 container_name, blob_name_prefix = blob_prefix.split("/", 1)

#                 blob_service_client = BlobServiceClient.from_connection_string(connection_string)
#                 container_client = blob_service_client.get_container_client(container_name)
                
#                 # List blobs under the prefix and download them
#                 blobs = container_client.list_blobs(name_starts_with=blob_name_prefix)
#                 for blob in blobs:
#                     local_file_path = os.path.join(temp_dir, os.path.relpath(blob.name, blob_name_prefix))
#                     os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
#                     blob_client = container_client.get_blob_client(blob)
#                     with open(local_file_path, "wb") as file:
#                         file.write(blob_client.download_blob().readall())

#                 # Move downloaded files to the designated local directory using shutil.move
#                 if not os.path.exists(local_dir):
#                     os.makedirs(local_dir)
#                 for filename in os.listdir(temp_dir):
#                     shutil.move(os.path.join(temp_dir, filename), os.path.join(local_dir, filename))

#                 # Load FAISS index from the local directory
#                 try:
#                     index = FAISS.load_local(
#                         folder_path=local_dir,
#                         index_name='index',
#                         embeddings=embeddings,
#                         allow_dangerous_deserialization=True
#                     )
#                     indexes[account].append(index)
#                 except Exception as e:
#                     print(f"Error loading index from {local_dir} for account {account}: {e}")

#     return indexes

# def search_across_indexes(vector_stores: List[FAISS], query: str):
#     all_results = []
#     for store in vector_stores:
#         results = store.similarity_search(query, fetch_k=8, k=2)
#         all_results.extend(results)
#     return all_results

# def display_messages():
#     if "messages" in st.session_state:
#         for message in st.session_state.messages:
#             with st.chat_message(message["role"]):
#                 st.write(message["content"])

# # Main chatbot function
# def run_chatbot(faiss_indexes, blob_service_client, embeddings, qa_chain):
#     with st.sidebar:
#         client_names = ["Select an Account Name"] + list(faiss_indexes.keys())
#         selected_client = st.selectbox("**Select Account Name** 🚩", client_names)#, index=client_names.index(st.session_state.get('clientOrg', "Select an Account Name")))
#         if selected_client != st.session_state['previous_clientOrg']:
#             st.session_state['clientOrg'] = selected_client
#             st.session_state['previous_clientOrg'] = selected_client
#             if st.session_state['clientOrg'] and st.session_state['clientOrg'] != "Select an Account Name":
#                 st.session_state['vector_store'] = faiss_indexes[st.session_state['clientOrg']]
#                 st.session_state["messages"] = list(st.session_state["default_messages"])
#                 st.session_state["show_feedback_form"] = False
#                 st.info(f"You are now connected to {st.session_state['clientOrg']} Account!")
#             else:
#                 st.warning("Add client name above")

#     memory = ConversationBufferMemory(
#         chat_memory=StreamlitChatMessageHistory(key="langchain_messages"),
#         return_messages=True,
#         memory_key="chat_history"
#     )

#     if st.sidebar.button("New Conversation"):
#         st.session_state["conversation_id"] = str(uuid.uuid4())
#         st.session_state["messages"] = list(st.session_state["default_messages"])
#         st.session_state["show_feedback_form"] = False
#         st.session_state["conversation_started"] = False
#         st.experimental_rerun()

#     display_messages()

#     user_prompt = st.chat_input("Type your message here")

#     if user_prompt:
#         st.session_state["user_prompt"] = user_prompt
#         st.session_state.messages.append({"role": "user", "content": user_prompt})
#         with st.chat_message("user"):
#             st.write(user_prompt)

#         input_dict = {"input": user_prompt}
#         try:
#             with collect_runs() as cb:
#                 if st.session_state.messages[-1]["role"] != "assistant":
#                     with st.chat_message("assistant"):
#                         with st.spinner("Generating answer..."):
#                             try:
#                                 vector_store = st.session_state.get('vector_store')
#                                 if not vector_store:
#                                     st.error("No Account Name Selected")
#                                 else:
#                                     relevant_docs = search_across_indexes(vector_store, user_prompt)

#                                     context = ""
#                                     relevant_images = []

#                                     for d in relevant_docs:
#                                         if d.metadata['type'] == 'text':
#                                             context += '[text]' + d.metadata['original_content']
#                                         elif d.metadata['type'] == 'table':
#                                             context += '[table]' + d.metadata['original_content']
#                                         elif d.metadata['type'] == 'image':
#                                             context += '[image]' + d.page_content
#                                             relevant_images.append(d.metadata['original_content'])

#                                     conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state['messages']])
#                                     context = conversation_history + "\n\n" + context

#                                     feedback_df = load_feedback(blob_service_client)
#                                     feedback_df = calculate_similarity_scores(feedback_df, user_prompt)
#                                     feedback_summary = analyze_feedback(feedback_df)
#                                     relevant_feedback_comments = enhance_response_with_feedback("", feedback_df, similarity_threshold=48)

#                                     initial_response = qa_chain.run({'context': context, 'question': user_prompt})
#                                     st.session_state["ai_response"] = enhance_response_with_feedback(initial_response, feedback_df, similarity_threshold=48)

#                                     st.write(st.session_state["ai_response"])
#                                     logger.info("AI response generated successfully.")

#                                     if relevant_images:
#                                         st.subheader("Relevant Images:")
#                                         cols = st.columns(len(relevant_images))
#                                         for idx, img in enumerate(relevant_images):
#                                             if isinstance(img, str):
#                                                 try:
#                                                     img_bytes = base64.b64decode(img)
#                                                     cols[idx].image(img_bytes, use_column_width=True, width=270)
#                                                 except Exception as e:
#                                                     logger.exception("Error decoding and displaying image: %s", e)
#                                                     st.error("Error decoding and displaying image. Please try again.")
#                             except Exception as e:
#                                 logger.exception("Error during processing: %s", e)
#                                 st.error("Connection aborted.")
#             new_ai_message = {"role": "assistant", "content": st.session_state["ai_response"]}
#             st.session_state.messages.append(new_ai_message)
#             if cb.traced_runs:
#                 st.session_state.run_id = cb.traced_runs[0].id
#             memory.save_context(input_dict, {"output": st.session_state["ai_response"]})
#             logger.info("Session state updated and context saved successfully.")
#             st.session_state["show_feedback_form"] = True

#             if not st.session_state["conversation_started"]:
#                 title = st.session_state["messages"][1]["content"][:30] + '...' if len(st.session_state["messages"][1]["content"]) > 30 else st.session_state["messages"][1]["content"]
#                 st.session_state["conversation_history"].insert(0, {
#                     "title": title,
#                     "messages": st.session_state["messages"],
#                     "conversation_id": st.session_state["conversation_id"]
#                 })
#                 st.session_state["conversation_started"] = True

#                 if len(st.session_state["conversation_history"]) > 5:
#                     st.session_state["conversation_history"] = st.session_state["conversation_history"][:5]
#         except Exception as e:
#             logger.exception("Error during the collection of runs or session state update: %s", e)
#             st.error("Connection aborted.")

#     if st.session_state["show_feedback_form"]:
#         st.subheader("Feedback")

#         rating_options = {
#             1: "Inaccurate",
#             2: "Partially Accurate",
#             3: "Accurate"
#         }

#         cols = st.columns(len(rating_options))
#         for idx, (rating, label) in enumerate(rating_options.items()):
#             if cols[idx].button(label):
#                 st.session_state["rating"] = rating

#         st.write(f"Selected Rating: {st.session_state['rating']}")

#         comments = st.text_area("Additional comments")
#         feedback_submitted = st.button("Submit Feedback")

#         if feedback_submitted:
#             if not comments:
#                 comments = "No comments"
#             feedback_data = {
#                 "conversation_id": st.session_state["conversation_id"],
#                 "username": st.session_state["username"],
#                 "account_name": st.session_state["clientOrg"],
#                 "conversation": st.session_state["messages"],
#                 "prompt": st.session_state["user_prompt"],
#                 "rating": st.session_state["rating"],
#                 "comments": comments
#             }
#             save_feedback(feedback_data, blob_service_client)
#             st.session_state["show_feedback_form"] = False
#             st.session_state["rating"] = 0
#             st.session_state["user_prompt"] = ""
#             st.session_state["ai_response"] = ""
#             st.success("Feedback submitted successfully")

#       # Conversation history with trash icon button
#     st.sidebar.markdown("""<div style="height: 1vh;"></div>""", unsafe_allow_html=True)

#     col1, col2 = st.sidebar.columns([4, 2])
#     with col1:
#         st.markdown("""<h3 style='font-size: 18px;'>Conversation History</h3>""", unsafe_allow_html=True)
#     with col2:
#         if st.button("🗑️", key="clear_button"):
#             st.session_state["conversation_history"] = []
#             st.session_state["conversation_deleted"] = True
#             st.experimental_rerun()

#     # Display conversation history buttons
#     for i, conversation in enumerate(st.session_state["conversation_history"]):
#         if st.sidebar.button(conversation["title"], key=f"conversation_{i}"):
#             st.session_state["selected_conversation"] = conversation
#             st.session_state["messages"] = conversation["messages"]
#             st.session_state["conversation_id"] = conversation["conversation_id"]
#             st.session_state["show_feedback_form"] = False
#             st.experimental_rerun()

#     # Show success message if conversations are deleted
#     if "conversation_deleted" in st.session_state and st.session_state["conversation_deleted"]:
#         st.sidebar.success("Conversations deleted!")
#         del st.session_state["conversation_deleted"]




import streamlit as st
import base64
import uuid
import logging
import shutil 
from typing import List, Dict
import pandas as pd
from azure.storage.blob import BlobServiceClient
from streamlit_extras.colored_header import colored_header
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.manager import collect_runs
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import os
import tempfile

# Set up logging
logger = logging.getLogger(__name__)

# Functions for Chatbot Module

def initialize_session_state():
    session_vars = {
        'clientOrg': 'Select an Account Name',
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
        'conversation_id': None,
        'selected_option': "Home",
        'faiss_indexes_loaded': False,  
    }
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default
    logger.info("Initialized session state")

def save_feedback(feedback_data, blob_service_client):
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

def load_feedback(blob_service_client, account_name: str):
    feedback_blob_name = "feedback.csv"
    feedback_blob_client = blob_service_client.get_blob_client(container="itgluecopilot", blob=feedback_blob_name)
    try:
        feedback_data = feedback_blob_client.download_blob().readall().decode('utf-8')
        feedback_df = pd.read_csv(io.StringIO(feedback_data))
        feedback_df = feedback_df[feedback_df['account_name'] == account_name]  # Filter by account
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

    user_prompt_processed = preprocess(user_prompt)
    feedback_df['combined_text'] = feedback_df.apply(lambda row: preprocess(row['conversation']) + ' ' + preprocess(row['comments']), axis=1)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([user_prompt_processed] + feedback_df['combined_text'].tolist())

    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    feedback_df['similarity'] = cosine_similarities * 100

    return feedback_df

def enhance_response_with_feedback(ai_response, feedback_df, similarity_threshold=48):
    filtered_feedback = feedback_df[feedback_df['similarity'] >= similarity_threshold]
    feedback_comments = filtered_feedback['comments'].tolist()

    if feedback_comments:
        enhanced_response = f"{ai_response}\n\nBased on user feedback, considering the following points:\n" + "\n".join([f"- {comment}" for comment in feedback_comments])
    else:
        enhanced_response = ai_response

    return enhanced_response

#LOADING INDEXES USING BLOB 

# def load_faiss_indexes(account_indexes: Dict[str, List[str]], embeddings, connection_string: str) -> Dict[str, List[FAISS]]:
#     indexes = {}
#     local_dir_base = "local_faiss_indexes"  # Base directory for local storage

#     if not os.path.exists(local_dir_base):
#         os.makedirs(local_dir_base)

#     for account, blob_prefixes in account_indexes.items():
#         indexes[account] = []
#         for blob_prefix in blob_prefixes:
#             local_dir = os.path.join(local_dir_base, account, os.path.basename(blob_prefix))

#             if os.path.exists(local_dir) and os.listdir(local_dir):
#                 try:
#                     index = FAISS.load_local(
#                         folder_path=local_dir,
#                         index_name='index',
#                         embeddings=embeddings,
#                         allow_dangerous_deserialization=True
#                     )
#                     indexes[account].append(index)
#                 except Exception as e:
#                     print(f"Error loading index from {local_dir} for account {account}: {e}")
#             else:
#                 temp_dir = tempfile.mkdtemp()
#                 container_name, blob_name_prefix = blob_prefix.split("/", 1)

#                 blob_service_client = BlobServiceClient.from_connection_string(connection_string)
#                 container_client = blob_service_client.get_container_client(container_name)
                
#                 blobs = container_client.list_blobs(name_starts_with=blob_name_prefix)
#                 for blob in blobs:
#                     local_file_path = os.path.join(temp_dir, os.path.relpath(blob.name, blob_name_prefix))
#                     os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
#                     blob_client = container_client.get_blob_client(blob)
#                     with open(local_file_path, "wb") as file:
#                         file.write(blob_client.download_blob().readall())

#                 if not os.path.exists(local_dir):
#                     os.makedirs(local_dir)
#                 for filename in os.listdir(temp_dir):
#                     shutil.move(os.path.join(temp_dir, filename), os.path.join(local_dir, filename))

#                 try:
#                     index = FAISS.load_local(
#                         folder_path=local_dir,
#                         index_name='index',
#                         embeddings=embeddings,
#                         allow_dangerous_deserialization=True
#                     )
#                     indexes[account].append(index)
#                 except Exception as e:
#                     print(f"Error loading index from {local_dir} for account {account}: {e}")

#     return indexes

#Using Github Local 

def load_faiss_indexes(account_indexes: Dict[str, List[str]],embeddings) -> Dict[str, List[FAISS]]:
    indexes = {}
    for account, paths in account_indexes.items():
        indexes[account] = []
        for path in paths:
            try:
                print(f"Loading index from: {path}")  
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
    ],
    "iBAS": [
        r"./Faiss_Index_IT Glue/Index_iBAS/index1",
        r"./Faiss_Index_IT Glue/Index_iBAS/index2_ocr"
    ]
}

def search_across_indexes(vector_stores: List[FAISS], query: str):
    all_results = []
    for store in vector_stores:
        results = store.similarity_search(query, fetch_k=8, k=2)
        all_results.extend(results)
    return all_results

def display_messages():
    if "messages" in st.session_state:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

def run_chatbot(faiss_indexes, blob_service_client, embeddings, qa_chain):
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

                                    feedback_df = load_feedback(blob_service_client, st.session_state['clientOrg'])  # Filter by account
                                    feedback_df = calculate_similarity_scores(feedback_df, user_prompt)
                                    feedback_summary = analyze_feedback(feedback_df)
                                    relevant_feedback_comments = enhance_response_with_feedback("", feedback_df, similarity_threshold=48)

                                    initial_response = qa_chain.run({'context': context, 'question': user_prompt})
                                    st.session_state["ai_response"] = enhance_response_with_feedback(initial_response, feedback_df, similarity_threshold=48)

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
                                st.error("Connection aborted.")
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
            st.error("Connection aborted.")

    if st.session_state["show_feedback_form"]:
        st.subheader("Feedback")

        rating_options = {
            1: "Inaccurate",
            2: "Partially Accurate",
            3: "Accurate"
        }

        cols = st.columns(len(rating_options))
        for idx, (rating, label) in enumerate(rating_options.items()):
            if cols[idx].button(label):
                st.session_state["rating"] = rating

        st.write(f"Selected Rating: {st.session_state['rating']}")

        comments = st.text_area("Additional comments")
        feedback_submitted = st.button("Submit Feedback")

        if feedback_submitted:
            if not comments:
                comments = "No comments"
            feedback_data = {
                "conversation_id": st.session_state["conversation_id"],
                "username": st.session_state["username"],
                "account_name": st.session_state["clientOrg"],
                "conversation": st.session_state["messages"],
                "prompt": st.session_state["user_prompt"],
                "rating": st.session_state["rating"],
                "comments": comments
            }
            save_feedback(feedback_data, blob_service_client)
            st.session_state["show_feedback_form"] = False
            st.session_state["rating"] = 0
            st.session_state["user_prompt"] = ""
            st.session_state["ai_response"] = ""
            st.success("Feedback submitted successfully")

    # Conversation history with trash icon button
    st.sidebar.markdown("""<div style="height: 1vh;"></div>""", unsafe_allow_html=True)

    col1, col2 = st.sidebar.columns([4, 2])
    with col1:
        st.markdown("""<h3 style='font-size: 18px;'>Conversation History</h3>""", unsafe_allow_html=True)
    with col2:
        if st.button("🗑️", key="clear_button"):
            st.session_state["conversation_history"] = []
            st.session_state["conversation_deleted"] = True
            st.experimental_rerun()

    # Display conversation history buttons
    for i, conversation in enumerate(st.session_state["conversation_history"]):
        if st.sidebar.button(conversation["title"], key=f"conversation_{i}"):
            st.session_state["selected_conversation"] = conversation
            st.session_state["messages"] = conversation["messages"]
            st.session_state["conversation_id"] = conversation["conversation_id"]
            st.session_state["show_feedback_form"] = False
            st.experimental_rerun()

    # Show success message if conversations are deleted
    if "conversation_deleted" in st.session_state and st.session_state["conversation_deleted"]:
        st.sidebar.success("Conversations deleted!")
        del st.session_state["conversation_deleted"]
