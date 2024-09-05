
def load_content_from_blob(blob_service_client, container_name, account_name, content_type):
    """Load specific content from a .txt file in Azure Blob Storage based on the account name and content type.
    If the file does not exist, create an empty one and upload it to Blob Storage."""
    blob_name = f"Documents/{account_name}_{content_type}.txt"

    try:
        blob_client = blob_service_client.get_blob_client(container_name, blob_name)
        blob_data = blob_client.download_blob().readall().decode('utf-8')
        return blob_data
    except Exception as e:
        logger.error(f"Error loading {content_type} for {account_name}: {e}. Creating a new empty file.")
        # If the file does not exist, create an empty one
        empty_content = ""
        try:
            blob_client.upload_blob(empty_content.encode('utf-8'), overwrite=True)
            logger.info(f"Created an empty {content_type} file for {account_name}.")
            return empty_content
        except Exception as upload_error:
            logger.error(f"Failed to create an empty {content_type} file for {account_name}: {upload_error}")
            return None

def save_content_to_blob(blob_service_client, container_name, account_name, content_type, content):
    """Save specific content to a .txt file in Azure Blob Storage based on the account name and content type."""
    blob_name = f"Documents/{account_name}_{content_type}.txt"
    try:
        blob_client = blob_service_client.get_blob_client(container_name, blob_name)
        blob_client.upload_blob(content.encode('utf-8'), overwrite=True)
        return True
    except Exception as e:
        logger.error(f"Error saving {content_type} for {account_name}: {e}")
        return False

def create_custom_prompt_template(escalation_matrix_content):
    """Creates a custom prompt template incorporating the escalation matrix content specific to the account."""
    return PromptTemplate.from_template(f"""
    ### Instruction ###
    Given the context below, provide a detailed and accurate answer to the question. Base your response primarily on the provided information. Only include additional context from external sources if absolutely necessary, and clearly identify it as such. DO NOT PARAPHRASE ANYTHING AND GIVE EXACTLY AS ITS GIVEN IN THE DOCUMENT.

    **Context:**
    {{context}}

    **Escalation Matrix:**
    {escalation_matrix_content}

    **Question:**
    {{question}}

    ### Guidelines ###
    1. **Primary Source**: Base your response primarily on the provided context and give the process as exact as given in the document.
    2. **External Information**: If additional information is needed, clearly label it as "External Information" and keep it to a minimum.
    3. **Specificity**: Provide detailed and precise information directly related to the query.
    4. **Separation of Information**: Use headings such as "IT Glue Response" and "External Information" to differentiate the sources.
    5. **Insufficient Context**: If the provided context does not contain enough information to answer the question, state: "The provided context does not contain enough information to answer the question."
    6. **Document References**: List the names of all documents accessed, along with a confidence score for each document based on its relevance. 
    7. Use the Escalation Matrix to help answer queries when necessary

    ### Example ###

    #### IT Glue Response ####
    [Your answer based on the given context]

    ## External Information ##
    [Your answer after not finding anything in context]
    
    ### Document Names ###
    [List of documents and confidence scores (in %) with descending order.] 
    
    **Answer:** 
    """)


# Import necessary libraries
import os
from azure.storage.blob import BlobServiceClient
import streamlit as st
from yaml.loader import SafeLoader
import yaml
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit_authenticator as stauth
from streamlit_navigation_bar import st_navbar
import logging
import pyotp
import qrcode
import io
import nltk

from chatbot import run_chatbot, initialize_session_state, load_faiss_indexes


nltk.download('punkt')
nltk.download('stopwords')

# Set page configuration
st.set_page_config(page_title="AI Support Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="auto")

# Navbar at the top
styles = {
    "span": {
        "border-radius": "0.1rem",
        "color": "orange",
        "margin": "0 0.125rem",
        "padding": "0.400rem 0.400rem",
    },
    "active": {
        "background-color": "rgba(255, 255, 255, 0.25)",
        "color": "orange",  # Active text color to orange
        "text-decoration": "underline",  # Underline the active text
    },
}

selected = st_navbar(["Home", "Chatbot", "Alerts and Escalation Matrix"], 
                     selected=st.session_state.get('selected_option', 'Home'), 
                     styles=styles)

# Initialize session state
initialize_session_state()

# Load environment variables
load_dotenv()

# Load config from Azure Blob Storage
connection_string = os.getenv("BLOB_CONNECTION_STRING")
container_name = "itgluecopilot"
config_blob_name = "config/config2.yaml"

# BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

# Load the YAML configuration file
blob_client = container_client.get_blob_client(config_blob_name)
blob_data = blob_client.download_blob().readall()
config = yaml.load(io.BytesIO(blob_data), Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Update session state with the selected option
st.session_state.selected_option = selected

with st.sidebar:
    st.image(r"./synoptek.png", width=275)

# Authentication for App
with st.sidebar:
    name, authentication_status, username = authenticator.login('Login', 'main')

if st.session_state["authentication_status"]:
    # Load account names dynamically from the blob
    account_names = ["Mitsui Chemicals America", "Northpoint Commercial Finance", "iBAS"]

    # Get user role after authentication
    username = st.session_state["username"]
    user_data = config['credentials']['usernames'][username]
    user_role = user_data.get('role', 'viewer')  # Default to 'viewer' if no role is specified
    st.session_state['user_role'] = user_role

    # Check for OTP Secret and Generate if Not Present
    otp_secret = user_data.get('otp_secret', "")

    if not otp_secret:
        otp_secret = pyotp.random_base32()
        config['credentials']['usernames'][username]['otp_secret'] = otp_secret
        blob_client.upload_blob(yaml.dump(config), overwrite=True)
        st.session_state['otp_setup_complete'] = False
        st.session_state['show_qr_code'] = True
        logger.info("Generated new OTP secret and set show_qr_code to True for user %s", username)
    else:
        st.session_state['otp_setup_complete'] = True

    # Ensure OTP secret is properly handled
    if otp_secret:
        totp = pyotp.TOTP(otp_secret)
        logger.info("Using OTP secret for user %s: %s", username, otp_secret)

        if not st.session_state['otp_verified']:
            if st.session_state['show_qr_code']:
                st.title("Welcome to AI Support Assistant! 👋")
                otp_uri = totp.provisioning_uri(name=user_data.get('email', ''), issuer_name="AI Support Assistant")
                qr = qrcode.make(otp_uri)
                qr = qr.resize((200, 200))

                st.image(qr, caption="Scan this QR code with your authenticator app")

            # st.title("AI Support Assistant")
            otp_input = st.text_input("Enter the OTP from your authenticator app", type="password")
            verify_button_clicked = st.button("Verify OTP")

            if verify_button_clicked:
                if totp.verify(otp_input):
                    st.session_state['otp_verified'] = True
                    st.session_state['show_qr_code'] = False
                    blob_client.upload_blob(yaml.dump(config), overwrite=True)
                    st.experimental_rerun()
                else:
                    st.error("Invalid OTP. Please try again.")
        else:
            # Sidebar account selection - Only show after successful OTP verification
            with st.sidebar:
                # st.image(r"./synoptek.png", width=275)
                
                client_names = ["Select an Account Name"] + account_names
                selected_client = st.selectbox("**Select Account Name** 🚩", client_names, key="sidebar_account_select")

                # Check if account selection has changed
                if selected_client != st.session_state.get('clientOrg', 'Select an Account Name'):
                    st.session_state['clientOrg'] = selected_client
                    st.session_state['previous_clientOrg'] = selected_client
                    st.experimental_rerun()  # Force rerun to synchronize selection

                # Load alerts and escalation content dynamically after account selection
                if st.session_state['clientOrg'] and st.session_state['clientOrg'] != "Select an Account Name":
                    st.info(f"You are viewing the {st.session_state['clientOrg']} Account")
                    alerts_content = load_content_from_blob(blob_service_client, container_name, st.session_state['clientOrg'], "Alerts")
                    escalation_matrix_content = load_content_from_blob(blob_service_client, container_name, st.session_state['clientOrg'], "Escalation Matrix")
                else:
                    alerts_content = None
                    escalation_matrix_content = None

            # Load FAISS indexes
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
                ],
            }
            embeddings = AzureOpenAIEmbeddings(
                azure_deployment='embeddings-aims',
                openai_api_version="2024-04-01-preview",
                azure_endpoint=os.getenv("OPENAI_ENDPOINT_AZURE"),
                api_key=os.getenv("OPENAI_API_KEY_AZURE")
            )
            faiss_indexes = load_faiss_indexes(account_indexes, embeddings)

            # --- APP ---
            if selected == "Home":
                st.write("# Welcome to AI Support Assistant! 👋")
                st.markdown(
                """
                Welcome to the AI Support Assistant! This tool is designed to streamline your support process by providing quick access to essential information and AI-powered assistance.

                ### Getting Started:

                - **👈 Select Account Name** and head to the "Chatbot" tab from the navigation bar to begin interacting with the AI Support Assistant.
                - Make sure to select the correct account in the Alerts and Escalation Matrix tab to view the most relevant information.

                ### How to Use the App:

                **1. Chatbot Tab:**  
                Navigate to the **Chatbot** tab to interact with our AI Assistant. You can ask questions or provide prompts related to your support needs, and the AI will generate detailed responses based on the context provided. This feature is ideal for quickly resolving issues or getting specific information.

                **Steps:**
                - Click on the **Chatbot** tab in the navigation bar and select an **Account**.
                - Type your question or request in the input box at the bottom of the page.
                - The AI will process your query and provide a response based on the available data.

                **2. Alerts and Escalation Matrix Tab:**  
                Visit the **Alerts and Escalation Matrix** tab to view critical alerts and the escalation matrix for specific accounts. This section provides important information about who to contact and the appropriate escalation procedures.

                **Steps:**
                - Click on the **Alerts and Escalation Matrix** tab in the navigation bar.
                - Use the sidebar to select the account you wish to view.
                - The page will display the relevant alerts and escalation matrix for the selected account.
                """
                )
            elif selected == 'Chatbot':
                if st.session_state['clientOrg'] and st.session_state['clientOrg'] != "Select an Account Name":
                    # Create a custom prompt template with the loaded escalation matrix content
                    custom_prompt_template = create_custom_prompt_template(escalation_matrix_content)

                    # Initialize LLMChain with the custom prompt template
                    qa_chain = LLMChain(
                        llm=AzureChatOpenAI(
                            model_name='gpt-4o',
                            openai_api_key=os.getenv("OPENAI_API_KEY_AZURE"),
                            azure_endpoint=os.getenv("OPENAI_ENDPOINT_AZURE"),
                            openai_api_version="2024-04-01-preview",
                            temperature=0,
                            max_tokens=4000,
                            streaming=True,
                            verbose=True,
                            model_kwargs={'seed': 123}
                        ),
                        prompt=custom_prompt_template
                    )

                    # Run the chatbot with the updated chain
                    run_chatbot(faiss_indexes, blob_service_client, embeddings, qa_chain)
                else:
                    st.warning("Please select an account name to get started")

            elif selected == 'Alerts and Escalation Matrix':
                # Since account selection is handled in the sidebar, no need for additional selectbox here
                if st.session_state['clientOrg'] and st.session_state['clientOrg'] != "Select an Account Name":
                    # Display or Edit Alerts Content
                    st.subheader("Alerts")
                    if st.session_state.get('editing_alerts', False):
                        edited_alerts_content = st.text_area("Edit Alerts", value=alerts_content if alerts_content else "No alerts content found. You can add new content.", height=500)
                        if st.button("Save Alerts Content", key="save_alerts_button"):
                            success = save_content_to_blob(blob_service_client, container_name, st.session_state['clientOrg'], "Alerts", edited_alerts_content)
                            if success:
                                st.success("Alerts content saved successfully!")
                                st.session_state['editing_alerts'] = False
                            else:
                                st.error("Failed to save alerts content.")
                            st.experimental_rerun()  # Force rerun after saving
                        if st.button("Cancel", key="cancel_alerts_button"):
                            st.session_state['editing_alerts'] = False
                            st.experimental_rerun()  # Force rerun after canceling
                    else:
                        if alerts_content:
                            st.markdown(
                                f"""
                                <div style="
                                    border: 2px solid #ffcc00; 
                                    padding: 15px; 
                                    border-radius: 10px;
                                    color:red; 
                                    background-color: #fff7e6;">
                                    {alerts_content}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.warning("No alerts content found. You can add new content.")
                        
                        # Show edit button only for allowed roles
                        if st.session_state['user_role'] in ['admin', 'editor']:
                            if st.button("Edit Alerts Content", key="edit_alerts_button"):
                                st.session_state['editing_alerts'] = True
                                st.experimental_rerun()

                    # Display or Edit Escalation Matrix Content
                    st.subheader("Escalation Matrix")
                    if st.session_state.get('editing_escalation_matrix', False):
                        edited_escalation_matrix_content = st.text_area("Edit Escalation Matrix", value=escalation_matrix_content if escalation_matrix_content else "No escalation matrix content found. You can add new content.", height=500)
                        if st.button("Save Escalation Matrix Content", key="save_matrix_button"):
                            success = save_content_to_blob(blob_service_client, container_name, st.session_state['clientOrg'], "Escalation Matrix", edited_escalation_matrix_content)
                            if success:
                                st.success("Escalation Matrix content saved successfully!")
                                st.session_state['editing_escalation_matrix'] = False
                            else:
                                st.error("Failed to save escalation matrix content.")
                            st.experimental_rerun()  # Force rerun after saving
                        if st.button("Cancel", key="cancel_matrix_button"):
                            st.session_state['editing_escalation_matrix'] = False
                            st.experimental_rerun()  # Force rerun after canceling
                    else:
                        if escalation_matrix_content:
                            st.markdown(
                                f"""
                                <div style="
                                    border: 2px solid #0066cc; 
                                    padding: 15px; 
                                    border-radius: 10px; 
                                    color: red;
                                    background-color: #e6f2ff;">
                                    {escalation_matrix_content}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.warning("No escalation matrix content found. You can add new content.")
                        
                        # Show edit button only for allowed roles
                        if st.session_state['user_role'] in ['admin', 'editor']:
                            if st.button("Edit Escalation Matrix Content", key="edit_matrix_button"):
                                st.session_state['editing_escalation_matrix'] = True
                                st.experimental_rerun()
                else:
                    st.subheader("Alerts")
                    st.warning("Please select an account name to view the alerts.")

                    st.subheader("Escalation Matrix")
                    st.warning("Please select an account name to view the escalation matrix.")

            st.sidebar.markdown("""<div style="height: 12vh;"></div>""", unsafe_allow_html=True)
            st.sidebar.markdown(f'## Hello, *{st.session_state["name"]}*')
            if st.sidebar.button("Logout", key="logout_button"):
                authenticator.logout('Logout', 'sidebar')
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.session_state['authentication_status'] = None
                st.experimental_rerun()

elif st.session_state["authentication_status"] == False:
    st.sidebar.error('Username/password is incorrect')
    st.write("# Welcome to AI Support Assistant! 👋")
    st.markdown(
        """
        Please enter your username and password to log in.
        """
    )
elif st.session_state["authentication_status"] == None:
    st.sidebar.warning('Please enter your username and password')
    st.write("# Welcome to AI Support Assistant! 👋")
    st.markdown(
        """
        Please enter your username and password to log in.
        """
    )
