# # # # # # # # import os
# # # # # # # # import tempfile
# # # # # # # # from azure.storage.blob import BlobServiceClient
# # # # # # # # import streamlit as st
# # # # # # # # from yaml.loader import SafeLoader
# # # # # # # # import yaml
# # # # # # # # from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
# # # # # # # # from langchain.chains import LLMChain
# # # # # # # # from langchain.prompts import PromptTemplate
# # # # # # # # from dotenv import load_dotenv
# # # # # # # # import streamlit_authenticator as stauth
# # # # # # # # from streamlit_navigation_bar import st_navbar
# # # # # # # # from azure.storage.blob import BlobServiceClient
# # # # # # # # import logging
# # # # # # # # import pyotp
# # # # # # # # import qrcode
# # # # # # # # import io

# # # # # # # # from aisupport import run_chatbot, initialize_session_state, load_faiss_indexes


# # # # # # # # # Set page configuration
# # # # # # # # st.set_page_config(page_title="AI Support Assistant", page_icon="🤖", layout="wide", initial_sidebar_state= "auto")

# # # # # # # # st.markdown(
# # # # # # # #     """
# # # # # # # #     <style>
# # # # # # # #     /* Ensures that the sidebar starts at the top */
# # # # # # # #     .css-1lcbmhc {
# # # # # # # #         padding-top: 0px;
# # # # # # # #     }
# # # # # # # #     /* Adjusts padding around the sidebar's content */
# # # # # # # #     .css-1aumxhk {
# # # # # # # #         padding-top: 0px;
# # # # # # # #     }
# # # # # # # #     </style>
# # # # # # # #     """,
# # # # # # # #     unsafe_allow_html=True
# # # # # # # # )
# # # # # # # # with st.sidebar:
# # # # # # # #        st.image(r"./synoptek.png", width=275)

# # # # # # # # load_dotenv()
# # # # # # # # # Load config
# # # # # # # # connection_string = os.getenv("BLOB_CONNECTION_STRING")
# # # # # # # # container_name = "itgluecopilot"
# # # # # # # # blob_name = "config/config.yaml"

# # # # # # # # # BlobServiceClient
# # # # # # # # blob_service_client = BlobServiceClient.from_connection_string(connection_string)
# # # # # # # # container_client = blob_service_client.get_container_client(container_name)

# # # # # # # # # Blob content to stream
# # # # # # # # blob_client = container_client.get_blob_client(blob_name)
# # # # # # # # blob_data = blob_client.download_blob().readall()

# # # # # # # # # Load the YAML
# # # # # # # # config = yaml.load(io.BytesIO(blob_data), Loader=SafeLoader)

# # # # # # # # authenticator = stauth.Authenticate(
# # # # # # # #     config['credentials'],
# # # # # # # #     config['cookie']['name'],
# # # # # # # #     config['cookie']['key'],
# # # # # # # #     config['cookie']['expiry_days'],
# # # # # # # # )

# # # # # # # # # Configure logging
# # # # # # # # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
# # # # # # # # logger = logging.getLogger(__name__)

# # # # # # # # # Load environment variables
# # # # # # # # load_dotenv()
# # # # # # # # logger.info("Environment variables loaded")

# # # # # # # # # Initialize session state
# # # # # # # # initialize_session_state()

# # # # # # # # # Authentication for App
# # # # # # # # with st.sidebar:
# # # # # # # #     name, authentication_status, username = authenticator.login('Login', 'main')

# # # # # # # # if st.session_state["authentication_status"]:
# # # # # # # #     # Check for OTP Secret and Generate if Not Present
# # # # # # # #     user_data = config['credentials']['usernames'].get(username, {})
# # # # # # # #     otp_secret = user_data.get('otp_secret', "")

# # # # # # # #     if not otp_secret:
# # # # # # # #         otp_secret = pyotp.random_base32()
# # # # # # # #         config['credentials']['usernames'][username]['otp_secret'] = otp_secret
# # # # # # # #         blob_client.upload_blob(yaml.dump(config), overwrite=True)
# # # # # # # #         st.session_state['otp_setup_complete'] = False
# # # # # # # #         st.session_state['show_qr_code'] = True
# # # # # # # #         logger.info("Generated new OTP secret and set show_qr_code to True for user %s", username)
# # # # # # # #     else:
# # # # # # # #         st.session_state['otp_setup_complete'] = True

# # # # # # # #     # Ensure OTP secret is properly handled
# # # # # # # #     if otp_secret:
# # # # # # # #         totp = pyotp.TOTP(otp_secret)
# # # # # # # #         logger.info("Using OTP secret for user %s: %s", username, otp_secret)

# # # # # # # #         if not st.session_state['otp_verified']:
# # # # # # # #             if st.session_state['show_qr_code']:
# # # # # # # #                 st.title("Welcome to AI Support Assistant! 👋")
# # # # # # # #                 otp_uri = totp.provisioning_uri(name=user_data.get('email', ''), issuer_name="AI Support Assistant")
# # # # # # # #                 qr = qrcode.make(otp_uri)
# # # # # # # #                 qr = qr.resize((200, 200))

# # # # # # # #                 st.image(qr, caption="Scan this QR code with your authenticator app")


# # # # # # # #             st.title("AI Support Assistant")
# # # # # # # #             otp_input = st.text_input("Enter the OTP from your authenticator app", type="password")
# # # # # # # #             verify_button_clicked = st.button("Verify OTP")

# # # # # # # #             if verify_button_clicked:
# # # # # # # #                 if totp.verify(otp_input):
# # # # # # # #                     st.session_state['otp_verified'] = True
# # # # # # # #                     st.session_state['show_qr_code'] = False
# # # # # # # #                     blob_client.upload_blob(yaml.dump(config), overwrite=True)
# # # # # # # #                     st.experimental_rerun()
# # # # # # # #                 else:
# # # # # # # #                     st.error("Invalid OTP. Please try again.")
# # # # # # # #         else:
# # # # # # # #             # Load FAISS indexes
# # # # # # # #             account_indexes = {
# # # # # # # #                 "Mitsui Chemicals America": [
# # # # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_Mitsui Chemicals America/index1",
# # # # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_Mitsui Chemicals America/index2_ocr"
# # # # # # # #                 ],
# # # # # # # #                 "Northpoint Commercial Finance": [
# # # # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_Northpoint Commercial Finance/index1",
# # # # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_Northpoint Commercial Finance/index2_ocr"
# # # # # # # #                 ],
# # # # # # # #                 "iBAS": [
# # # # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_iBAS/index1",
# # # # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_iBAS/index2_ocr"
# # # # # # # #                 ]
# # # # # # # #             }
# # # # # # # #             embeddings = AzureOpenAIEmbeddings(
# # # # # # # #                 azure_deployment='embeddings-aims',
# # # # # # # #                 openai_api_version="2024-04-01-preview",
# # # # # # # #                 azure_endpoint=os.getenv("OPENAI_ENDPOINT_AZURE"),
# # # # # # # #                 api_key=os.getenv("OPENAI_API_KEY_AZURE")
# # # # # # # #             )
# # # # # # # #             faiss_indexes = load_faiss_indexes(account_indexes, embeddings, connection_string)

# # # # # # # #             # Navigation and Main Content
# # # # # # # #             styles = {
# # # # # # # #                 "span": {
# # # # # # # #                     "border-radius": "0.1rem",
# # # # # # # #                     "color": "rgb(49, 51, 63)",
# # # # # # # #                     "margin": "0 0.125rem",
# # # # # # # #                     "padding": "0.400rem 0.400rem",
# # # # # # # #                 },
# # # # # # # #                 "active": {
# # # # # # # #                     "background-color": "rgba(255, 255, 255, 0.25)",
# # # # # # # #                 },
# # # # # # # #             }

# # # # # # # #             selected = st_navbar(["Home", "Chatbot", "Alerts and Escalation Matrix"],
# # # # # # # #                                  selected=st.session_state.selected_option, styles=styles)

# # # # # # # #             # --- APP ---
# # # # # # # #             if selected == "Home":
# # # # # # # #                 st.write("# Welcome to AI Support Assistant! 👋")
# # # # # # # #                 st.markdown(
# # # # # # # #                     """
# # # # # # # #                     Welcome to our AI Support Assistant. 
# # # # # # # #                     Use this tool to streamline your support process.
                    
# # # # # # # #                     **☝️ Select an option from the navigation bar** to get started!

# # # # # # # #                     Please head to the Chatbot Tab to get started with your queries.
# # # # # # # #                     """
# # # # # # # #                 )
# # # # # # # #                 st.session_state.selected_option = 'Home'
# # # # # # # #             elif selected == 'Chatbot':
# # # # # # # #                 qa_chain = LLMChain(llm=AzureChatOpenAI(
# # # # # # # #                     model_name='gpt-4o',
# # # # # # # #                     openai_api_key=os.getenv("OPENAI_API_KEY_AZURE"),
# # # # # # # #                     azure_endpoint=os.getenv("OPENAI_ENDPOINT_AZURE"),
# # # # # # # #                     openai_api_version="2024-04-01-preview",
# # # # # # # #                     temperature=0,
# # # # # # # #                     max_tokens=4000,
# # # # # # # #                     streaming=True,
# # # # # # # #                     verbose=True,
# # # # # # # #                     model_kwargs={'seed': 123}
# # # # # # # #                 ), prompt=PromptTemplate.from_template("""
# # # # # # # #                 ### Instruction ###
# # # # # # # #                 Given the context below, provide a detailed and accurate answer to the question. Base your response primarily on the provided information. Only include additional context from external sources if absolutely necessary, and clearly identify it as such. DO NOT PARAPHRASE ANYTHING AND GIVE EXACTLY AS ITS GIVEN IN THE DOCUMENT.

# # # # # # # #                 **Context:**
# # # # # # # #                 {context}

# # # # # # # #                 **Question:**
# # # # # # # #                 {question}

# # # # # # # #                 ### Guidelines ###
# # # # # # # #                 1. **Primary Source**: Base your response primarily on the provided context and give the process as exact as given in the document.
# # # # # # # #                 2. **External Information**: If additional information is needed, clearly label it as "External Information" and keep it to a minimum.
# # # # # # # #                 3. **Specificity**: Provide detailed and precise information directly related to the query.
# # # # # # # #                 4. **Separation of Information**: Use headings such as "IT Glue Response" and "External Information" to differentiate the sources.
# # # # # # # #                 5. **Insufficient Context**: If the provided context does not contain enough information to answer the question, state: "The provided context does not contain enough information to answer the question."
# # # # # # # #                 6. **Document References**: List the names of all documents accessed, along with a confidence score for each document based on its relevance. 

# # # # # # # #                 ### Example ###

# # # # # # # #                 #### IT Glue Response ####
# # # # # # # #                 [Your answer based on the given context]
                                                       
# # # # # # # #                 ## External Information ##
# # # # # # # #                 []

# # # # # # # #                 #### Alerts and Escalation Matrix ####
# # # # # # # #                 [Your answer after referring to the specific escalation matrix file for the given account name to determine who should be alerted or to whom the issue should be escalated. Ensure that the appropriate contacts are notified based on the escalation matrix provided in the document.] 
                
# # # # # # # #                 ### Document Names ###
# # # # # # # #                 [List of documents and confidence scores (in %) with descending order.] 
                
# # # # # # # #                 **Answer:**
# # # # # # # #                 """))
# # # # # # # #                 run_chatbot(faiss_indexes, blob_service_client, embeddings, qa_chain)
# # # # # # # #                 st.session_state.selected_option = 'Chatbot'

# # # # # # # #             elif selected == 'Alerts and Escalation Matrix':

# # # # # # # #                 with st.sidebar:
# # # # # # # #                     client_names = ["Select an Account Name"] + list(faiss_indexes.keys())
# # # # # # # #                     selected_client = st.selectbox("**Select Account Name** 🚩", client_names)#, index=client_names.index(st.session_state.get('clientOrg', "Select an Account Name")))
# # # # # # # #                     if selected_client != st.session_state['previous_clientOrg']:
# # # # # # # #                         st.session_state['clientOrg'] = selected_client
# # # # # # # #                         st.session_state['previous_clientOrg'] = selected_client
# # # # # # # #                         if st.session_state['clientOrg'] and st.session_state['clientOrg'] != "Select an Account Name":
# # # # # # # #                             st.info(f"You are now viewing {st.session_state['clientOrg']} Account!")
# # # # # # # #                         else:
# # # # # # # #                             st.warning("Please select an account name above.")

# # # # # # # #                 if st.session_state['clientOrg'] == "Mitsui Chemicals America":
# # # # # # # #                     st.write("# Mitsui Chemicals America Alerts and Escalation Matrix")
# # # # # # # #                     st.markdown("Instructions specific to Mitsui Chemicals America...")
# # # # # # # #                 elif st.session_state['clientOrg'] == "Northpoint Commercial Finance":
# # # # # # # #                     st.write("# Northpoint Commercial Finance Alerts and Escalation Matrix")
# # # # # # # #                     st.markdown("Instructions specific to Northpoint Commercial Finance...")
# # # # # # # #                 elif st.session_state['clientOrg'] == "iBAS":
# # # # # # # #                     st.write("# iBAS Alerts and Escalation Matrix")
# # # # # # # #                     st.markdown('''Alerts for iBAS or CHCS
                                
# # # # # # # #                     iBAS may also be referred to as CHCS or Teleo. The parent company is Teleo, which created the holding company iBAS to contain their acquisition, CHCS.

# # # # # # # #             AS400/TS4300 tape change requests are to be set to P2, and GSSD is to notify the team lead when they are received to be worked.

# # # # # # # #             ***All change requests need to be reviewed in the weekly CAB that CHCS Services hosts on Wednesday mornings at 7:30 a.m. (mountain time). 
# # # # # # # #             Add the original requester in the description of the CR; Requester: name of requester. Add Paul Miller as approver on the change. 
# # # # # # # #             Peer approval needs to be done by Tuesday at noon Mountain. Once all CHCS approvals are given, Paul will give the final approval in the CR. 
# # # # # # # #             Add ticket notes when the change is complete, as that will be audited by CHCS. 
# # # # # # # #             *** https://synoptek.itglue.com/5000748/docs/11688669#version=published&documentMode=view***

# # # # # # # #             If a call center employee is down, this is an immediate P2, and their client SLAs are impacted.
# # # # # # # #             End users are primarily east coast US or Noida India, and most work remotely from home.
# # # # # # # #             Access to work applications is primarily through RDS (Remote Desktop Services).

# # # # # # # #             Documentation is here: https://synoptek.itglue.com/5000748/documents/folder/3025208/


# # # # # # # #             Synoptek Support Number:
# # # # # # # #             US: 877-796-2310
# # # # # # # #             India: 1800-309-8033
# # # # # # # #             ***EOC please add ftpservices@chcs-services.com to any FTP server alerts***

# # # # # # # #             Ticket Detail Alert
# # # # # # # #             iBAS may also be referred to as CHCS or Teleo. The parent company is Teleo, which created the holding company iBAS to contain their acquisition, CHCS. 
# # # # # # # #             AS400/TS4300 tape change requests are to be set to P2, and GSSD is to notify the team lead when they are received to be worked. 
# # # # # # # #             ***All change requests need to be reviewed in the weekly CAB that CHCS Services hosts on Wednesday mornings at 7:30 a.m. (mountain time). 
# # # # # # # #             Add the original requester in the description of the CR; Requester: name of requester. Add Paul Miller as approver on the change. Peer approval needs to be done by Tuesday at noon Mountain. Once all CHCS approvals are given, Paul will give the final approval in the CR. Add ticket notes when the change is complete, as that will be audited by CHCS. *** https://synoptek.itglue.com/5000748/docs/11688669#version=published&documentMode=view If a call center employee is down, this is an immediate P2, and their client SLAs are impacted. End users are primarily east coast US or Noida India, and most work remotely from home. Access to work applications is primarily through RDS (Remote Desktop Services). Documentation is here: https://synoptek.itglue.com/5000748/documents/folder/3025208/ Synoptek Support Number: -US: 877-796-2310 -India: 1800-309-8033 ***EOC please add ftpservices@chcs-services.com to any FTP server alerts***

# # # # # # # #             Customer Deployment Status
# # # # # # # #             THE CUSTOMER IS CURRENTLY IN DEPLOYMENT WITH TRANSITIONAL SUPPORT. ACTIVE DEPLOY PROJECTS LISTED BELOW:
# # # # # # # #             No Active Deployment Projects
# # # # # # # #             Bulletin Board
# # # # # # # #             No Critical Tickets


# # # # # # # #             Support Info	-
# # # # # # # #             Client Delivery Team-	CDS-Northwest
# # # # # # # #             Client Advisor-	MICHELLE CARROLL
# # # # # # # #             Sales Rep	
# # # # # # # #             Client Delivery Manager- REED, MIKE
# # # # # # # #             Client Delivery Lead-	REED, MIKE
                                
# # # # # # # #             ''')
# # # # # # # #                     st.markdown('''Escalation Matrix
                                
                                
                                
# # # # # # # #                                 ''')
# # # # # # # #                 else:
# # # # # # # #                     st.write("# Alerts and Escalation Matrix")
# # # # # # # #                     st.markdown("Please select an account to view the specific Alerts and Escalation Matrix.")
# # # # # # # #                 st.session_state.selected_option = 'Alerts and Escalation Matrix'

# # # # # # # #             st.sidebar.markdown("""<div style="height: 12vh;"></div>""", unsafe_allow_html=True)
# # # # # # # #             st.sidebar.markdown(f'## Hello, *{st.session_state["name"]}*')
# # # # # # # #             if st.sidebar.button("Logout", key="logout_button"):
# # # # # # # #                 authenticator.logout('Logout', 'sidebar')
# # # # # # # #                 for key in list(st.session_state.keys()):
# # # # # # # #                     del st.session_state[key]
# # # # # # # #                 st.experimental_rerun()

# # # # # # # # elif st.session_state["authentication_status"] == False:
# # # # # # # #     st.sidebar.error('Username/password is incorrect')
# # # # # # # # elif st.session_state["authentication_status"] == None:
# # # # # # # #     st.sidebar.warning('Please enter your username and password')


# # # # # # # import os
# # # # # # # import tempfile
# # # # # # # from azure.storage.blob import BlobServiceClient
# # # # # # # import streamlit as st
# # # # # # # from yaml.loader import SafeLoader
# # # # # # # import yaml
# # # # # # # from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
# # # # # # # from langchain.chains import LLMChain
# # # # # # # from langchain.prompts import PromptTemplate
# # # # # # # from dotenv import load_dotenv
# # # # # # # import streamlit_authenticator as stauth
# # # # # # # from streamlit_navigation_bar import st_navbar
# # # # # # # import logging
# # # # # # # import pyotp
# # # # # # # import qrcode
# # # # # # # import io

# # # # # # # from aisupport import run_chatbot, initialize_session_state, load_faiss_indexes

# # # # # # # # Set page configuration
# # # # # # # st.set_page_config(page_title="AI Support Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="auto")

# # # # # # # st.markdown(
# # # # # # #     """
# # # # # # #     <style>
# # # # # # #     /* Ensures that the sidebar starts at the top */
# # # # # # #     .css-1lcbmhc {
# # # # # # #         padding-top: 0px;
# # # # # # #     }
# # # # # # #     /* Adjusts padding around the sidebar's content */
# # # # # # #     .css-1aumxhk {
# # # # # # #         padding-top: 0px;
# # # # # # #     }
# # # # # # #     </style>
# # # # # # #     """,
# # # # # # #     unsafe_allow_html=True
# # # # # # # )
# # # # # # # with st.sidebar:
# # # # # # #     st.image(r"./synoptek.png", width=275)

# # # # # # # load_dotenv()
# # # # # # # # Load config
# # # # # # # connection_string = os.getenv("BLOB_CONNECTION_STRING")
# # # # # # # container_name = "itgluecopilot"
# # # # # # # config_blob_name = "config/config.yaml"
# # # # # # # accounts_blob_name = "config/accounts.txt"

# # # # # # # # BlobServiceClient
# # # # # # # blob_service_client = BlobServiceClient.from_connection_string(connection_string)
# # # # # # # container_client = blob_service_client.get_container_client(container_name)

# # # # # # # # Load the YAML configuration file
# # # # # # # blob_client = container_client.get_blob_client(config_blob_name)
# # # # # # # blob_data = blob_client.download_blob().readall()
# # # # # # # config = yaml.load(io.BytesIO(blob_data), Loader=SafeLoader)

# # # # # # # authenticator = stauth.Authenticate(
# # # # # # #     config['credentials'],
# # # # # # #     config['cookie']['name'],
# # # # # # #     config['cookie']['key'],
# # # # # # #     config['cookie']['expiry_days'],
# # # # # # # )

# # # # # # # # Configure logging
# # # # # # # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
# # # # # # # logger = logging.getLogger(__name__)

# # # # # # # # Load environment variables
# # # # # # # load_dotenv()
# # # # # # # logger.info("Environment variables loaded")

# # # # # # # # Initialize session state
# # # # # # # initialize_session_state()

# # # # # # # # Authentication for App
# # # # # # # with st.sidebar:
# # # # # # #     name, authentication_status, username = authenticator.login('Login', 'main')

# # # # # # # def load_content_from_blob(blob_service_client, container_name, account_name, content_type):
# # # # # # #     """Load specific content from a .txt file in Azure Blob Storage based on the account name and content type."""
# # # # # # #     blob_name = f"Documents/{account_name}/{content_type}.txt"

# # # # # # #     try:
# # # # # # #         blob_client = blob_service_client.get_blob_client(container_name, blob_name)
# # # # # # #         blob_data = blob_client.download_blob().readall().decode('utf-8')
# # # # # # #         return blob_data
# # # # # # #     except Exception as e:
# # # # # # #         logger.error(f"Error loading {content_type} for {account_name}: {e}")
# # # # # # #         return f"Error loading {content_type} content for {account_name}."

# # # # # # # if st.session_state["authentication_status"]:
# # # # # # #     # Load account names dynamically from the blob
# # # # # # #     account_names = ["Mitsui Chemicals America", "Northpoint Commercial Finance", "iBAS"]

# # # # # # #     # Check for OTP Secret and Generate if Not Present
# # # # # # #     user_data = config['credentials']['usernames'].get(username, {})
# # # # # # #     otp_secret = user_data.get('otp_secret', "")

# # # # # # #     if not otp_secret:
# # # # # # #         otp_secret = pyotp.random_base32()
# # # # # # #         config['credentials']['usernames'][username]['otp_secret'] = otp_secret
# # # # # # #         blob_client.upload_blob(yaml.dump(config), overwrite=True)
# # # # # # #         st.session_state['otp_setup_complete'] = False
# # # # # # #         st.session_state['show_qr_code'] = True
# # # # # # #         logger.info("Generated new OTP secret and set show_qr_code to True for user %s", username)
# # # # # # #     else:
# # # # # # #         st.session_state['otp_setup_complete'] = True

# # # # # # #     # Ensure OTP secret is properly handled
# # # # # # #     if otp_secret:
# # # # # # #         totp = pyotp.TOTP(otp_secret)
# # # # # # #         logger.info("Using OTP secret for user %s: %s", username, otp_secret)

# # # # # # #         if not st.session_state['otp_verified']:
# # # # # # #             if st.session_state['show_qr_code']:
# # # # # # #                 st.title("Welcome to AI Support Assistant! 👋")
# # # # # # #                 otp_uri = totp.provisioning_uri(name=user_data.get('email', ''), issuer_name="AI Support Assistant")
# # # # # # #                 qr = qrcode.make(otp_uri)
# # # # # # #                 qr = qr.resize((200, 200))

# # # # # # #                 st.image(qr, caption="Scan this QR code with your authenticator app")

# # # # # # #             st.title("AI Support Assistant")
# # # # # # #             otp_input = st.text_input("Enter the OTP from your authenticator app", type="password")
# # # # # # #             verify_button_clicked = st.button("Verify OTP")

# # # # # # #             if verify_button_clicked:
# # # # # # #                 if totp.verify(otp_input):
# # # # # # #                     st.session_state['otp_verified'] = True
# # # # # # #                     st.session_state['show_qr_code'] = False
# # # # # # #                     blob_client.upload_blob(yaml.dump(config), overwrite=True)
# # # # # # #                     st.experimental_rerun()
# # # # # # #                 else:
# # # # # # #                     st.error("Invalid OTP. Please try again.")
# # # # # # #         else:
# # # # # # #             # Load FAISS indexes
# # # # # # #             account_indexes = {
# # # # # # #                 "Mitsui Chemicals America": [
# # # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_Mitsui Chemicals America/index1",
# # # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_Mitsui Chemicals America/index2_ocr"
# # # # # # #                 ],
# # # # # # #                 "Northpoint Commercial Finance": [
# # # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_Northpoint Commercial Finance/index1",
# # # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_Northpoint Commercial Finance/index2_ocr"
# # # # # # #                 ],
# # # # # # #                 "iBAS": [
# # # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_iBAS/index1",
# # # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_iBAS/index2_ocr"
# # # # # # #                 ]
# # # # # # #             }
# # # # # # #             embeddings = AzureOpenAIEmbeddings(
# # # # # # #                 azure_deployment='embeddings-aims',
# # # # # # #                 openai_api_version="2024-04-01-preview",
# # # # # # #                 azure_endpoint=os.getenv("OPENAI_ENDPOINT_AZURE"),
# # # # # # #                 api_key=os.getenv("OPENAI_API_KEY_AZURE")
# # # # # # #             )
# # # # # # #             faiss_indexes = load_faiss_indexes(account_indexes, embeddings, connection_string)

# # # # # # #             # Navigation and Main Content
# # # # # # #             styles = {
# # # # # # #                 "span": {
# # # # # # #                     "border-radius": "0.1rem",
# # # # # # #                     "color": "rgb(49, 51, 63)",
# # # # # # #                     "margin": "0 0.125rem",
# # # # # # #                     "padding": "0.400rem 0.400rem",
# # # # # # #                 },
# # # # # # #                 "active": {
# # # # # # #                     "background-color": "rgba(255, 255, 255, 0.25)",
# # # # # # #                 },
# # # # # # #             }

# # # # # # #             selected = st_navbar(["Home", "Chatbot", "Alerts and Escalation Matrix"],
# # # # # # #                                  selected=st.session_state.selected_option, styles=styles)

# # # # # # #             # --- APP ---
# # # # # # #             if selected == "Home":
# # # # # # #                 st.write("# Welcome to AI Support Assistant! 👋")
# # # # # # #                 st.markdown(
# # # # # # #                     """
# # # # # # #                     Welcome to our AI Support Assistant. 
# # # # # # #                     Use this tool to streamline your support process.
                    
# # # # # # #                     **☝️ Select an option from the navigation bar** to get started!

# # # # # # #                     Please head to the Chatbot Tab to get started with your queries.
# # # # # # #                     """
# # # # # # #                 )
# # # # # # #                 st.session_state.selected_option = 'Home'
# # # # # # #             elif selected == 'Chatbot':
# # # # # # #                 qa_chain = LLMChain(llm=AzureChatOpenAI(
# # # # # # #                     model_name='gpt-4o',
# # # # # # #                     openai_api_key=os.getenv("OPENAI_API_KEY_AZURE"),
# # # # # # #                     azure_endpoint=os.getenv("OPENAI_ENDPOINT_AZURE"),
# # # # # # #                     openai_api_version="2024-04-01-preview",
# # # # # # #                     temperature=0,
# # # # # # #                     max_tokens=4000,
# # # # # # #                     streaming=True,
# # # # # # #                     verbose=True,
# # # # # # #                     model_kwargs={'seed': 123}
# # # # # # #                 ), prompt=PromptTemplate.from_template("""
# # # # # # #                 ### Instruction ###
# # # # # # #                 Given the context below, provide a detailed and accurate answer to the question. Base your response primarily on the provided information. Only include additional context from external sources if absolutely necessary, and clearly identify it as such. DO NOT PARAPHRASE ANYTHING AND GIVE EXACTLY AS ITS GIVEN IN THE DOCUMENT.

# # # # # # #                 **Context:**
# # # # # # #                 {context}

# # # # # # #                 **Question:**
# # # # # # #                 {question}

# # # # # # #                 ### Guidelines ###
# # # # # # #                 1. **Primary Source**: Base your response primarily on the provided context and give the process as exact as given in the document.
# # # # # # #                 2. **External Information**: If additional information is needed, clearly label it as "External Information" and keep it to a minimum.
# # # # # # #                 3. **Specificity**: Provide detailed and precise information directly related to the query.
# # # # # # #                 4. **Separation of Information**: Use headings such as "IT Glue Response" and "External Information" to differentiate the sources.
# # # # # # #                 5. **Insufficient Context**: If the provided context does not contain enough information to answer the question, state: "The provided context does not contain enough information to answer the question."
# # # # # # #                 6. **Document References**: List the names of all documents accessed, along with a confidence score for each document based on its relevance. 

# # # # # # #                 ### Example ###

# # # # # # #                 #### IT Glue Response ####
# # # # # # #                 [Your answer based on the given context]
                                                       
# # # # # # #                 ## External Information ##
# # # # # # #                 []

# # # # # # #                 #### Alerts and Escalation Matrix ####
# # # # # # #                 [Your answer after referring to the specific escalation matrix file for the given account name to determine who should be alerted or to whom the issue should be escalated. Ensure that the appropriate contacts are notified based on the escalation matrix provided in the document.] 
                
# # # # # # #                 ### Document Names ###
# # # # # # #                 [List of documents and confidence scores (in %) with descending order.] 
                
# # # # # # #                 **Answer:**
# # # # # # #                 """))
# # # # # # #                 run_chatbot(faiss_indexes, blob_service_client, embeddings, qa_chain)
# # # # # # #                 st.session_state.selected_option = 'Chatbot'

# # # # # # #             elif selected == 'Alerts and Escalation Matrix':

# # # # # # #                 with st.sidebar:
# # # # # # #                     client_names = ["Select an Account Name"] + account_names
# # # # # # #                     selected_client = st.selectbox("**Select Account Name** 🚩", client_names)
# # # # # # #                     if selected_client != st.session_state['previous_clientOrg']:
# # # # # # #                         st.session_state['clientOrg'] = selected_client
# # # # # # #                         st.session_state['previous_clientOrg'] = selected_client
# # # # # # #                         if st.session_state['clientOrg'] and st.session_state['clientOrg'] != "Select an Account Name":
# # # # # # #                             st.info(f"You are now viewing {st.session_state['clientOrg']} Account!")
# # # # # # #                             alerts_content = load_content_from_blob(blob_service_client, container_name, st.session_state['clientOrg'], "alerts")
# # # # # # #                             escalation_matrix_content = load_content_from_blob(blob_service_client, container_name, st.session_state['clientOrg'], "escalation_matrix")

# # # # # # #                             st.markdown("### Alerts")
# # # # # # #                             st.markdown(alerts_content)

# # # # # # #                             st.markdown("### Escalation Matrix")
# # # # # # #                             st.markdown(escalation_matrix_content)
# # # # # # #                         else:
# # # # # # #                             st.warning("Please select an account name above.")

# # # # # # #             st.sidebar.markdown("""<div style="height: 12vh;"></div>""", unsafe_allow_html=True)
# # # # # # #             st.sidebar.markdown(f'## Hello, *{st.session_state["name"]}*')
# # # # # # #             if st.sidebar.button("Logout", key="logout_button"):
# # # # # # #                 authenticator.logout('Logout', 'sidebar')
# # # # # # #                 for key in list(st.session_state.keys()):
# # # # # # #                     del st.session_state[key]
# # # # # # #                 st.experimental_rerun()

# # # # # # # elif st.session_state["authentication_status"] == False:
# # # # # # #     st.sidebar.error('Username/password is incorrect')
# # # # # # # elif st.session_state["authentication_status"] == None:
# # # # # # #     st.sidebar.warning('Please enter your username and password')



# # # # # # # import os
# # # # # # # from azure.storage.blob import BlobServiceClient
# # # # # # # import streamlit as st
# # # # # # # from yaml.loader import SafeLoader
# # # # # # # import yaml
# # # # # # # from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
# # # # # # # from langchain.chains import LLMChain
# # # # # # # from langchain.prompts import PromptTemplate
# # # # # # # from dotenv import load_dotenv
# # # # # # # import streamlit_authenticator as stauth
# # # # # # # from streamlit_navigation_bar import st_navbar
# # # # # # # import logging
# # # # # # # import pyotp
# # # # # # # import qrcode
# # # # # # # import io

# # # # # # # from aisupport import run_chatbot, initialize_session_state, load_faiss_indexes

# # # # # # # # Set page configuration
# # # # # # # st.set_page_config(page_title="AI Support Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="auto")

# # # # # # # st.markdown(
# # # # # # #     """
# # # # # # #     <style>
# # # # # # #     /* Ensures that the sidebar starts at the top */
# # # # # # #     .css-1lcbmhc {
# # # # # # #         padding-top: 0px;
# # # # # # #     }
# # # # # # #     /* Adjusts padding around the sidebar's content */
# # # # # # #     .css-1aumxhk {
# # # # # # #         padding-top: 0px;
# # # # # # #     }
# # # # # # #     </style>
# # # # # # #     """,
# # # # # # #     unsafe_allow_html=True
# # # # # # # )
# # # # # # # with st.sidebar:
# # # # # # #     st.image(r"./synoptek.png", width=275)

# # # # # # # load_dotenv()
# # # # # # # # Load config
# # # # # # # connection_string = os.getenv("BLOB_CONNECTION_STRING")
# # # # # # # container_name = "itgluecopilot"
# # # # # # # config_blob_name = "config/config.yaml"
# # # # # # # accounts_blob_name = "config/accounts.txt"

# # # # # # # # BlobServiceClient
# # # # # # # blob_service_client = BlobServiceClient.from_connection_string(connection_string)
# # # # # # # container_client = blob_service_client.get_container_client(container_name)

# # # # # # # # Load the YAML configuration file
# # # # # # # blob_client = container_client.get_blob_client(config_blob_name)
# # # # # # # blob_data = blob_client.download_blob().readall()
# # # # # # # config = yaml.load(io.BytesIO(blob_data), Loader=SafeLoader)

# # # # # # # authenticator = stauth.Authenticate(
# # # # # # #     config['credentials'],
# # # # # # #     config['cookie']['name'],
# # # # # # #     config['cookie']['key'],
# # # # # # #     config['cookie']['expiry_days'],
# # # # # # # )

# # # # # # # # Configure logging
# # # # # # # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levellevelname)s - %(message)s", handlers=[logging.StreamHandler()])
# # # # # # # logger = logging.getLogger(__name__)

# # # # # # # # Load environment variables
# # # # # # # load_dotenv()
# # # # # # # logger.info("Environment variables loaded")

# # # # # # # # Initialize session state
# # # # # # # initialize_session_state()

# # # # # # # # Authentication for App
# # # # # # # with st.sidebar:
# # # # # # #     name, authentication_status, username = authenticator.login('Login', 'main')

# # # # # # # def load_content_from_blob(blob_service_client, container_name, account_name, content_type):
# # # # # # #     """Load specific content from a .txt file in Azure Blob Storage based on the account name and content type."""
# # # # # # #     blob_name = f"Documents/{account_name}/{content_type}.txt"

# # # # # # #     try:
# # # # # # #         blob_client = blob_service_client.get_blob_client(container_name, blob_name)
# # # # # # #         blob_data = blob_client.download_blob().readall().decode('utf-8')
# # # # # # #         return blob_data
# # # # # # #     except Exception as e:
# # # # # # #         logger.error(f"Error loading {content_type} for {account_name}: {e}")
# # # # # # #         return f"Error: {content_type} content not found for {account_name}. Please ensure the file exists in Azure Blob Storage."

# # # # # # # if st.session_state["authentication_status"]:
# # # # # # #     # Load account names dynamically from the blob
# # # # # # #     account_names = ["Mitsui Chemicals America", "Northpoint Commercial Finance", "iBAS"]

# # # # # # #     # Check for OTP Secret and Generate if Not Present
# # # # # # #     user_data = config['credentials']['usernames'].get(username, {})
# # # # # # #     otp_secret = user_data.get('otp_secret', "")

# # # # # # #     if not otp_secret:
# # # # # # #         otp_secret = pyotp.random_base32()
# # # # # # #         config['credentials']['usernames'][username]['otp_secret'] = otp_secret
# # # # # # #         blob_client.upload_blob(yaml.dump(config), overwrite=True)
# # # # # # #         st.session_state['otp_setup_complete'] = False
# # # # # # #         st.session_state['show_qr_code'] = True
# # # # # # #         logger.info("Generated new OTP secret and set show_qr_code to True for user %s", username)
# # # # # # #     else:
# # # # # # #         st.session_state['otp_setup_complete'] = True

# # # # # # #     # Ensure OTP secret is properly handled
# # # # # # #     if otp_secret:
# # # # # # #         totp = pyotp.TOTP(otp_secret)
# # # # # # #         logger.info("Using OTP secret for user %s: %s", username, otp_secret)

# # # # # # #         if not st.session_state['otp_verified']:
# # # # # # #             if st.session_state['show_qr_code']:
# # # # # # #                 st.title("Welcome to AI Support Assistant! 👋")
# # # # # # #                 otp_uri = totp.provisioning_uri(name=user_data.get('email', ''), issuer_name="AI Support Assistant")
# # # # # # #                 qr = qrcode.make(otp_uri)
# # # # # # #                 qr = qr.resize((200, 200))

# # # # # # #                 st.image(qr, caption="Scan this QR code with your authenticator app")

# # # # # # #             st.title("AI Support Assistant")
# # # # # # #             otp_input = st.text_input("Enter the OTP from your authenticator app", type="password")
# # # # # # #             verify_button_clicked = st.button("Verify OTP")

# # # # # # #             if verify_button_clicked:
# # # # # # #                 if totp.verify(otp_input):
# # # # # # #                     st.session_state['otp_verified'] = True
# # # # # # #                     st.session_state['show_qr_code'] = False
# # # # # # #                     blob_client.upload_blob(yaml.dump(config), overwrite=True)
# # # # # # #                     st.experimental_rerun()
# # # # # # #                 else:
# # # # # # #                     st.error("Invalid OTP. Please try again.")
# # # # # # #         else:
# # # # # # #             # Load FAISS indexes
# # # # # # #             account_indexes = {
# # # # # # #                 "Mitsui Chemicals America": [
# # # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_Mitsui Chemicals America/index1",
# # # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_Mitsui Chemicals America/index2_ocr"
# # # # # # #                 ],
# # # # # # #                 "Northpoint Commercial Finance": [
# # # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_Northpoint Commercial Finance/index1",
# # # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_Northpoint Commercial Finance/index2_ocr"
# # # # # # #                 ],
# # # # # # #                 "iBAS": [
# # # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_iBAS/index1",
# # # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_iBAS/index2_ocr"
# # # # # # #                 ]
# # # # # # #             }
# # # # # # #             embeddings = AzureOpenAIEmbeddings(
# # # # # # #                 azure_deployment='embeddings-aims',
# # # # # # #                 openai_api_version="2024-04-01-preview",
# # # # # # #                 azure_endpoint=os.getenv("OPENAI_ENDPOINT_AZURE"),
# # # # # # #                 api_key=os.getenv("OPENAI_API_KEY_AZURE")
# # # # # # #             )
# # # # # # #             faiss_indexes = load_faiss_indexes(account_indexes, embeddings, connection_string)

# # # # # # #             # Navigation and Main Content
# # # # # # #             styles = {
# # # # # # #                 "span": {
# # # # # # #                     "border-radius": "0.1rem",
# # # # # # #                     "color": "rgb(49, 51, 63)",
# # # # # # #                     "margin": "0 0.125rem",
# # # # # # #                     "padding": "0.400rem 0.400rem",
# # # # # # #                 },
# # # # # # #                 "active": {
# # # # # # #                     "background-color": "rgba(255, 255, 255, 0.25)",
# # # # # # #                 },
# # # # # # #             }

# # # # # # #             selected = st_navbar(["Home", "Chatbot", "Alerts and Escalation Matrix"],
# # # # # # #                                  selected=st.session_state.selected_option, styles=styles)

# # # # # # #             # --- APP ---
# # # # # # #             if selected == "Home":
# # # # # # #                 st.write("# Welcome to AI Support Assistant! 👋")
# # # # # # #                 st.markdown(
# # # # # # #                     """
# # # # # # #                     Welcome to our AI Support Assistant. 
# # # # # # #                     Use this tool to streamline your support process.
                    
# # # # # # #                     **☝️ Select an option from the navigation bar** to get started!

# # # # # # #                     Please head to the Chatbot Tab to get started with your queries.
# # # # # # #                     """
# # # # # # #                 )
# # # # # # #                 st.session_state.selected_option = 'Home'
# # # # # # #             elif selected == 'Chatbot':
# # # # # # #                 qa_chain = LLMChain(llm=AzureChatOpenAI(
# # # # # # #                     model_name='gpt-4o',
# # # # # # #                     openai_api_key=os.getenv("OPENAI_API_KEY_AZURE"),
# # # # # # #                     azure_endpoint=os.getenv("OPENAI_ENDPOINT_AZURE"),
# # # # # # #                     openai_api_version="2024-04-01-preview",
# # # # # # #                     temperature=0,
# # # # # # #                     max_tokens=4000,
# # # # # # #                     streaming=True,
# # # # # # #                     verbose=True,
# # # # # # #                     model_kwargs={'seed': 123}
# # # # # # #                 ), prompt=PromptTemplate.from_template("""
# # # # # # #                 ### Instruction ###
# # # # # # #                 Given the context below, provide a detailed and accurate answer to the question. Base your response primarily on the provided information. Only include additional context from external sources if absolutely necessary, and clearly identify it as such. DO NOT PARAPHRASE ANYTHING AND GIVE EXACTLY AS ITS GIVEN IN THE DOCUMENT.

# # # # # # #                 **Context:**
# # # # # # #                 {context}

# # # # # # #                 **Question:**
# # # # # # #                 {question}

# # # # # # #                 ### Guidelines ###
# # # # # # #                 1. **Primary Source**: Base your response primarily on the provided context and give the process as exact as given in the document.
# # # # # # #                 2. **External Information**: If additional information is needed, clearly label it as "External Information" and keep it to a minimum.
# # # # # # #                 3. **Specificity**: Provide detailed and precise information directly related to the query.
# # # # # # #                 4. **Separation of Information**: Use headings such as "IT Glue Response" and "External Information" to differentiate the sources.
# # # # # # #                 5. **Insufficient Context**: If the provided context does not contain enough information to answer the question, state: "The provided context does not contain enough information to answer the question."
# # # # # # #                 6. **Document References**: List the names of all documents accessed, along with a confidence score for each document based on its relevance. 

# # # # # # #                 ### Example ###

# # # # # # #                 #### IT Glue Response ####
# # # # # # #                 [Your answer based on the given context]
                                                       
# # # # # # #                 ## External Information ##
# # # # # # #                 []

# # # # # # #                 #### Alerts and Escalation Matrix ####
# # # # # # #                 [Your answer after referring to the specific escalation matrix file for the given account name to determine who should be alerted or to whom the issue should be escalated. Ensure that the appropriate contacts are notified based on the escalation matrix provided in the document.] 
                
# # # # # # #                 ### Document Names ###
# # # # # # #                 [List of documents and confidence scores (in %) with descending order.] 
                
# # # # # # #                 **Answer:**
# # # # # # #                 """))
# # # # # # #                 run_chatbot(faiss_indexes, blob_service_client, embeddings, qa_chain)
# # # # # # #                 st.session_state.selected_option = 'Chatbot'

# # # # # # #             elif selected == 'Alerts and Escalation Matrix':
# # # # # # #                 with st.sidebar:
# # # # # # #                     client_names = ["Select an Account Name"] + account_names
# # # # # # #                     selected_client = st.selectbox("**Select Account Name** 🚩", client_names)
# # # # # # #                     if selected_client != st.session_state['previous_clientOrg']:
# # # # # # #                         st.session_state['clientOrg'] = selected_client
# # # # # # #                         st.session_state['previous_clientOrg'] = selected_client
                
# # # # # # #                     if st.session_state['clientOrg'] and st.session_state['clientOrg'] != "Select an Account Name":
# # # # # # #                         st.markdown(f"**You are viewing the {st.session_state['clientOrg']} Account**")
                  


# # # # # # #                 # if selected_client != st.session_state['previous_clientOrg']:
# # # # # # #                 #     st.session_state['clientOrg'] = selected_client
# # # # # # #                 #     st.session_state['previous_clientOrg'] = selected_client
# # # # # # #                 #     if st.session_state['clientOrg'] and st.session_state['clientOrg'] != "Select an Account Name":
# # # # # # #                 #         st.info(f"You are now viewing {st.session_state['clientOrg']} Account!")
                        
# # # # # # #                     alerts_content = load_content_from_blob(blob_service_client, container_name, st.session_state['clientOrg'], "Alerts")
# # # # # # #                     escalation_matrix_content = load_content_from_blob(blob_service_client, container_name, st.session_state['clientOrg'], "Escalation Matrix")

# # # # # # #                     st.subheader("Alerts")
# # # # # # #                     st.markdown(alerts_content)

# # # # # # #                     st.subheader("Escalation Matrix")
# # # # # # #                     st.markdown(escalation_matrix_content)

# # # # # # #                     else:
# # # # # # #                         st.warning("Please select an account name above.")

# # # # # # #             st.sidebar.markdown("""<div style="height: 12vh;"></div>""", unsafe_allow_html=True)
# # # # # # #             st.sidebar.markdown(f'## Hello, *{st.session_state["name"]}*')
# # # # # # #             if st.sidebar.button("Logout", key="logout_button"):
# # # # # # #                 authenticator.logout('Logout', 'sidebar')
# # # # # # #                 for key in list(st.session_state.keys()):
# # # # # # #                     del st.session_state[key]
# # # # # # #                 st.experimental_rerun()

# # # # # # # elif st.session_state["authentication_status"] == False:
# # # # # # #     st.sidebar.error('Username/password is incorrect')
# # # # # # # elif st.session_state["authentication_status"] == None:
# # # # # # #     st.sidebar.warning('Please enter your username and password')


# # # # # # # import os
# # # # # # # from azure.storage.blob import BlobServiceClient
# # # # # # # import streamlit as st
# # # # # # # from yaml.loader import SafeLoader
# # # # # # # import yaml
# # # # # # # from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
# # # # # # # from langchain.chains import LLMChain
# # # # # # # from langchain.prompts import PromptTemplate
# # # # # # # from dotenv import load_dotenv
# # # # # # # import streamlit_authenticator as stauth
# # # # # # # from streamlit_navigation_bar import st_navbar
# # # # # # # import logging
# # # # # # # import pyotp
# # # # # # # import qrcode
# # # # # # # import io

# # # # # # # from aisupport import run_chatbot, initialize_session_state, load_faiss_indexes

# # # # # # # # Set page configuration
# # # # # # # st.set_page_config(page_title="AI Support Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="auto")

# # # # # # # st.markdown(
# # # # # # #     """
# # # # # # #     <style>
# # # # # # #     /* Ensures that the sidebar starts at the top */
# # # # # # #     .css-1lcbmhc {
# # # # # # #         padding-top: 0px;
# # # # # # #     }
# # # # # # #     /* Adjusts padding around the sidebar's content */
# # # # # # #     .css-1aumxhk {
# # # # # # #         padding-top: 0px;
# # # # # # #     }
# # # # # # #     </style>
# # # # # # #     """,
# # # # # # #     unsafe_allow_html=True
# # # # # # # )

# # # # # # # with st.sidebar:
# # # # # # #     st.image(r"./synoptek.png", width=275)

# # # # # # # load_dotenv()
# # # # # # # # Load config
# # # # # # # connection_string = os.getenv("BLOB_CONNECTION_STRING")
# # # # # # # container_name = "itgluecopilot"
# # # # # # # config_blob_name = "config/config.yaml"
# # # # # # # accounts_blob_name = "config/accounts.txt"

# # # # # # # # BlobServiceClient
# # # # # # # blob_service_client = BlobServiceClient.from_connection_string(connection_string)
# # # # # # # container_client = blob_service_client.get_container_client(container_name)

# # # # # # # # Load the YAML configuration file
# # # # # # # blob_client = container_client.get_blob_client(config_blob_name)
# # # # # # # blob_data = blob_client.download_blob().readall()
# # # # # # # config = yaml.load(io.BytesIO(blob_data), Loader=SafeLoader)

# # # # # # # authenticator = stauth.Authenticate(
# # # # # # #     config['credentials'],
# # # # # # #     config['cookie']['name'],
# # # # # # #     config['cookie']['key'],
# # # # # # #     config['cookie']['expiry_days'],
# # # # # # # )

# # # # # # # # Configure logging
# # # # # # # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(level)s - %(message)s", handlers=[logging.StreamHandler()])
# # # # # # # logger = logging.getLogger(__name__)

# # # # # # # # Load environment variables
# # # # # # # load_dotenv()
# # # # # # # logger.info("Environment variables loaded")

# # # # # # # # Initialize session state
# # # # # # # initialize_session_state()

# # # # # # # # Authentication for App
# # # # # # # with st.sidebar:
# # # # # # #     name, authentication_status, username = authenticator.login('Login', 'main')

# # # # # # # def load_content_from_blob(blob_service_client, container_name, account_name, content_type):
# # # # # # #     """Load specific content from a .txt file in Azure Blob Storage based on the account name and content type."""
# # # # # # #     blob_name = f"Documents/{account_name}/{content_type}.txt"

# # # # # # #     try:
# # # # # # #         blob_client = blob_service_client.get_blob_client(container_name, blob_name)
# # # # # # #         blob_data = blob_client.download_blob().readall().decode('utf-8')
# # # # # # #         return blob_data
# # # # # # #     except Exception as e:
# # # # # # #         logger.error(f"Error loading {content_type} for {account_name}: {e}")
# # # # # # #         return f"Error: {content_type} content not found for {account_name}. Please ensure the file exists in Azure Blob Storage."

# # # # # # # if st.session_state["authentication_status"]:
# # # # # # #     # Load account names dynamically from the blob
# # # # # # #     account_names = ["Mitsui Chemicals America", "Northpoint Commercial Finance", "iBAS"]

# # # # # # #     # Check for OTP Secret and Generate if Not Present
# # # # # # #     user_data = config['credentials']['usernames'].get(username, {})
# # # # # # #     otp_secret = user_data.get('otp_secret', "")

# # # # # # #     if not otp_secret:
# # # # # # #         otp_secret = pyotp.random_base32()
# # # # # # #         config['credentials']['usernames'][username]['otp_secret'] = otp_secret
# # # # # # #         blob_client.upload_blob(yaml.dump(config), overwrite=True)
# # # # # # #         st.session_state['otp_setup_complete'] = False
# # # # # # #         st.session_state['show_qr_code'] = True
# # # # # # #         logger.info("Generated new OTP secret and set show_qr_code to True for user %s", username)
# # # # # # #     else:
# # # # # # #         st.session_state['otp_setup_complete'] = True

# # # # # # #     # Ensure OTP secret is properly handled
# # # # # # #     if otp_secret:
# # # # # # #         totp = pyotp.TOTP(otp_secret)
# # # # # # #         logger.info("Using OTP secret for user %s: %s", username, otp_secret)

# # # # # # #         if not st.session_state['otp_verified']:
# # # # # # #             if st.session_state['show_qr_code']:
# # # # # # #                 st.title("Welcome to AI Support Assistant! 👋")
# # # # # # #                 otp_uri = totp.provisioning_uri(name=user_data.get('email', ''), issuer_name="AI Support Assistant")
# # # # # # #                 qr = qrcode.make(otp_uri)
# # # # # # #                 qr = qr.resize((200, 200))

# # # # # # #                 st.image(qr, caption="Scan this QR code with your authenticator app")

# # # # # # #             st.title("AI Support Assistant")
# # # # # # #             otp_input = st.text_input("Enter the OTP from your authenticator app", type="password")
# # # # # # #             verify_button_clicked = st.button("Verify OTP")

# # # # # # #             if verify_button_clicked:
# # # # # # #                 if totp.verify(otp_input):
# # # # # # #                     st.session_state['otp_verified'] = True
# # # # # # #                     st.session_state['show_qr_code'] = False
# # # # # # #                     blob_client.upload_blob(yaml.dump(config), overwrite=True)
# # # # # # #                     st.experimental_rerun()
# # # # # # #                 else:
# # # # # # #                     st.error("Invalid OTP. Please try again.")
# # # # # # #         else:
# # # # # # #             # Load FAISS indexes
# # # # # # #             account_indexes = {
# # # # # # #                 "Mitsui Chemicals America": [
# # # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_Mitsui Chemicals America/index1",
# # # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_Mitsui Chemicals America/index2_ocr"
# # # # # # #                 ],
# # # # # # #                 "Northpoint Commercial Finance": [
# # # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_Northpoint Commercial Finance/index1",
# # # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_Northpoint Commercial Finance/index2_ocr"
# # # # # # #                 ],
# # # # # # #                 "iBAS": [
# # # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_iBAS/index1",
# # # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_iBAS/index2_ocr"
# # # # # # #                 ]
# # # # # # #             }
# # # # # # #             embeddings = AzureOpenAIEmbeddings(
# # # # # # #                 azure_deployment='embeddings-aims',
# # # # # # #                 openai_api_version="2024-04-01-preview",
# # # # # # #                 azure_endpoint=os.getenv("OPENAI_ENDPOINT_AZURE"),
# # # # # # #                 api_key=os.getenv("OPENAI_API_KEY_AZURE")
# # # # # # #             )
# # # # # # #             faiss_indexes = load_faiss_indexes(account_indexes, embeddings, connection_string)

# # # # # # #             # Navigation and Main Content
# # # # # # #             styles = {
# # # # # # #                 "span": {
# # # # # # #                     "border-radius": "0.1rem",
# # # # # # #                     "color": "rgb(49, 51, 63)",
# # # # # # #                     "margin": "0 0.125rem",
# # # # # # #                     "padding": "0.400rem 0.400rem",
# # # # # # #                 },
# # # # # # #                 "active": {
# # # # # # #                     "background-color": "rgba(255, 255, 255, 0.25)",
# # # # # # #                 },
# # # # # # #             }

# # # # # # #             selected = st_navbar(["Home", "Chatbot", "Alerts and Escalation Matrix"],
# # # # # # #                                  selected=st.session_state.selected_option, styles=styles)

# # # # # # #             # --- APP ---
# # # # # # #             if selected == "Home":
# # # # # # #                 st.write("# Welcome to AI Support Assistant! 👋")
# # # # # # #                 st.markdown(
# # # # # # #                     """
# # # # # # #                     Welcome to our AI Support Assistant. 
# # # # # # #                     Use this tool to streamline your support process.
                    
# # # # # # #                     **☝️ Select an option from the navigation bar** to get started!

# # # # # # #                     Please head to the Chatbot Tab to get started with your queries.
# # # # # # #                     """
# # # # # # #                 )
# # # # # # #                 st.session_state.selected_option = 'Home'
# # # # # # #             elif selected == 'Chatbot':
# # # # # # #                 qa_chain = LLMChain(llm=AzureChatOpenAI(
# # # # # # #                     model_name='gpt-4o',
# # # # # # #                     openai_api_key=os.getenv("OPENAI_API_KEY_AZURE"),
# # # # # # #                     azure_endpoint=os.getenv("OPENAI_ENDPOINT_AZURE"),
# # # # # # #                     openai_api_version="2024-04-01-preview",
# # # # # # #                     temperature=0,
# # # # # # #                     max_tokens=4000,
# # # # # # #                     streaming=True,
# # # # # # #                     verbose=True,
# # # # # # #                     model_kwargs={'seed': 123}
# # # # # # #                 ), prompt=PromptTemplate.from_template("""
# # # # # # #                 ### Instruction ###
# # # # # # #                 Given the context below, provide a detailed and accurate answer to the question. Base your response primarily on the provided information. Only include additional context from external sources if absolutely necessary, and clearly identify it as such. DO NOT PARAPHRASE ANYTHING AND GIVE EXACTLY AS ITS GIVEN IN THE DOCUMENT.

# # # # # # #                 **Context:**
# # # # # # #                 {context}

# # # # # # #                 **Question:**
# # # # # # #                 {question}

# # # # # # #                 ### Guidelines ###
# # # # # # #                 1. **Primary Source**: Base your response primarily on the provided context and give the process as exact as given in the document.
# # # # # # #                 2. **External Information**: If additional information is needed, clearly label it as "External Information" and keep it to a minimum.
# # # # # # #                 3. **Specificity**: Provide detailed and precise information directly related to the query.
# # # # # # #                 4. **Separation of Information**: Use headings such as "IT Glue Response" and "External Information" to differentiate the sources.
# # # # # # #                 5. **Insufficient Context**: If the provided context does not contain enough information to answer the question, state: "The provided context does not contain enough information to answer the question."
# # # # # # #                 6. **Document References**: List the names of all documents accessed, along with a confidence score for each document based on its relevance. 

# # # # # # #                 ### Example ###

# # # # # # #                 #### IT Glue Response ####
# # # # # # #                 [Your answer based on the given context]
                                                       
# # # # # # #                 ## External Information ##
# # # # # # #                 []

# # # # # # #                 #### Alerts and Escalation Matrix ####
# # # # # # #                 [Your answer after referring to the specific escalation matrix file for the given account name to determine who should be alerted or to whom the issue should be escalated. Ensure that the appropriate contacts are notified based on the escalation matrix provided in the document.] 
                
# # # # # # #                 ### Document Names ###
# # # # # # #                 [List of documents and confidence scores (in %) with descending order.] 
                
# # # # # # #                 **Answer:**
# # # # # # #                 """))
# # # # # # #                 run_chatbot(faiss_indexes, blob_service_client, embeddings, qa_chain)
# # # # # # #                 st.session_state.selected_option = 'Chatbot'

# # # # # # #             elif selected == 'Alerts and Escalation Matrix':
# # # # # # #                 with st.sidebar:
# # # # # # #                     client_names = ["Select an Account Name"] + account_names
# # # # # # #                     selected_client = st.selectbox("**Select Account Name** 🚩", client_names)
# # # # # # #                     if selected_client != st.session_state['previous_clientOrg']:
# # # # # # #                         st.session_state['clientOrg'] = selected_client
# # # # # # #                         st.session_state['previous_clientOrg'] = selected_client

# # # # # # #                     if st.session_state['clientOrg'] and st.session_state['clientOrg'] != "Select an Account Name":
# # # # # # #                         st.info(f"You are viewing the {st.session_state['clientOrg']} Account")

# # # # # # #                 if st.session_state['clientOrg'] and st.session_state['clientOrg'] != "Select an Account Name":
# # # # # # #                     alerts_content = load_content_from_blob(blob_service_client, container_name, st.session_state['clientOrg'], "Alerts")
# # # # # # #                     escalation_matrix_content = load_content_from_blob(blob_service_client, container_name, st.session_state['clientOrg'], "Escalation Matrix")

# # # # # # #                     st.subheader("Alerts")
# # # # # # #                     st.markdown(alerts_content)

# # # # # # #                     st.subheader("Escalation Matrix")
# # # # # # #                     st.markdown(escalation_matrix_content)

# # # # # # #                 else:
# # # # # # #                     st.warning("Please select an account name above.")

# # # # # # #             st.sidebar.markdown("""<div style="height: 12vh;"></div>""", unsafe_allow_html=True)
# # # # # # #             st.sidebar.markdown(f'## Hello, *{st.session_state["name"]}*')
# # # # # # #             if st.sidebar.button("Logout", key="logout_button"):
# # # # # # #                 authenticator.logout('Logout', 'sidebar')
# # # # # # #                 for key in list(st.session_state.keys()):
# # # # # # #                     del st.session_state[key]
# # # # # # #                 st.experimental_rerun()

# # # # # # # elif st.session_state["authentication_status"] == False:
# # # # # # #     st.sidebar.error('Username/password is incorrect')
# # # # # # # elif st.session_state["authentication_status"] == None:
# # # # # # #     st.sidebar.warning('Please enter your username and password')




# # # # # # import os
# # # # # # from azure.storage.blob import BlobServiceClient
# # # # # # import streamlit as st
# # # # # # from yaml.loader import SafeLoader
# # # # # # import yaml
# # # # # # from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
# # # # # # from langchain.chains import LLMChain
# # # # # # from langchain.prompts import PromptTemplate
# # # # # # from dotenv import load_dotenv
# # # # # # import streamlit_authenticator as stauth
# # # # # # from streamlit_navigation_bar import st_navbar
# # # # # # import logging
# # # # # # import pyotp
# # # # # # import qrcode
# # # # # # import io

# # # # # # from aisupport import run_chatbot, initialize_session_state, load_faiss_indexes

# # # # # # # Set page configuration
# # # # # # st.set_page_config(page_title="AI Support Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="auto")

# # # # # # st.markdown(
# # # # # #     """
# # # # # #     <style>
# # # # # #     /* Ensures that the sidebar starts at the top */
# # # # # #     .css-1lcbmhc {
# # # # # #         padding-top: 0px;
# # # # # #     }
# # # # # #     /* Adjusts padding around the sidebar's content */
# # # # # #     .css-1aumxhk {
# # # # # #         padding-top: 0px;
# # # # # #     }
# # # # # #     </style>
# # # # # #     """,
# # # # # #     unsafe_allow_html=True
# # # # # # )

# # # # # # with st.sidebar:
# # # # # #     st.image(r"./synoptek.png", width=275)

# # # # # # load_dotenv()
# # # # # # # Load config
# # # # # # connection_string = os.getenv("BLOB_CONNECTION_STRING")
# # # # # # container_name = "itgluecopilot"
# # # # # # config_blob_name = "config/config.yaml"
# # # # # # accounts_blob_name = "config/accounts.txt"

# # # # # # # BlobServiceClient
# # # # # # blob_service_client = BlobServiceClient.from_connection_string(connection_string)
# # # # # # container_client = blob_service_client.get_container_client(container_name)

# # # # # # # Load the YAML configuration file
# # # # # # blob_client = container_client.get_blob_client(config_blob_name)
# # # # # # blob_data = blob_client.download_blob().readall()
# # # # # # config = yaml.load(io.BytesIO(blob_data), Loader=SafeLoader)

# # # # # # authenticator = stauth.Authenticate(
# # # # # #     config['credentials'],
# # # # # #     config['cookie']['name'],
# # # # # #     config['cookie']['key'],
# # # # # #     config['cookie']['expiry_days'],
# # # # # # )

# # # # # # # Configure logging
# # # # # # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(level)s - %(message)s", handlers=[logging.StreamHandler()])
# # # # # # logger = logging.getLogger(__name__)

# # # # # # # Load environment variables
# # # # # # load_dotenv()
# # # # # # logger.info("Environment variables loaded")

# # # # # # # Initialize session state
# # # # # # initialize_session_state()

# # # # # # # Authentication for App
# # # # # # with st.sidebar:
# # # # # #     name, authentication_status, username = authenticator.login('Login', 'main')

# # # # # # def load_content_from_blob(blob_service_client, container_name, account_name, content_type):
# # # # # #     """Load specific content from a .txt file in Azure Blob Storage based on the account name and content type."""
# # # # # #     blob_name = f"Documents/{account_name}/{content_type}.txt"

# # # # # #     try:
# # # # # #         blob_client = blob_service_client.get_blob_client(container_name, blob_name)
# # # # # #         blob_data = blob_client.download_blob().readall().decode('utf-8')
# # # # # #         return blob_data
# # # # # #     except Exception as e:
# # # # # #         # logger.error(f"Error loading {content_type} for {account_name}: {e}")
# # # # # #         return f"{content_type} content not found for {account_name}. Please ensure the file exists."

# # # # # # if st.session_state["authentication_status"]:
# # # # # #     # Load account names dynamically from the blob
# # # # # #     account_names = ["Mitsui Chemicals America", "Northpoint Commercial Finance", "iBAS"]

# # # # # #     # Check for OTP Secret and Generate if Not Present
# # # # # #     user_data = config['credentials']['usernames'].get(username, {})
# # # # # #     otp_secret = user_data.get('otp_secret', "")

# # # # # #     if not otp_secret:
# # # # # #         otp_secret = pyotp.random_base32()
# # # # # #         config['credentials']['usernames'][username]['otp_secret'] = otp_secret
# # # # # #         blob_client.upload_blob(yaml.dump(config), overwrite=True)
# # # # # #         st.session_state['otp_setup_complete'] = False
# # # # # #         st.session_state['show_qr_code'] = True
# # # # # #         logger.info("Generated new OTP secret and set show_qr_code to True for user %s", username)
# # # # # #     else:
# # # # # #         st.session_state['otp_setup_complete'] = True

# # # # # #     # Ensure OTP secret is properly handled
# # # # # #     if otp_secret:
# # # # # #         totp = pyotp.TOTP(otp_secret)
# # # # # #         logger.info("Using OTP secret for user %s: %s", username, otp_secret)

# # # # # #         if not st.session_state['otp_verified']:
# # # # # #             if st.session_state['show_qr_code']:
# # # # # #                 st.title("Welcome to AI Support Assistant! 👋")
# # # # # #                 otp_uri = totp.provisioning_uri(name=user_data.get('email', ''), issuer_name="AI Support Assistant")
# # # # # #                 qr = qrcode.make(otp_uri)
# # # # # #                 qr = qr.resize((200, 200))

# # # # # #                 st.image(qr, caption="Scan this QR code with your authenticator app")

# # # # # #             st.title("AI Support Assistant")
# # # # # #             otp_input = st.text_input("Enter the OTP from your authenticator app", type="password")
# # # # # #             verify_button_clicked = st.button("Verify OTP")

# # # # # #             if verify_button_clicked:
# # # # # #                 if totp.verify(otp_input):
# # # # # #                     st.session_state['otp_verified'] = True
# # # # # #                     st.session_state['show_qr_code'] = False
# # # # # #                     blob_client.upload_blob(yaml.dump(config), overwrite=True)
# # # # # #                     st.experimental_rerun()
# # # # # #                 else:
# # # # # #                     st.error("Invalid OTP. Please try again.")
# # # # # #         else:
# # # # # #             # Load FAISS indexes
# # # # # #             account_indexes = {
# # # # # #                 "Mitsui Chemicals America": [
# # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_Mitsui Chemicals America/index1",
# # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_Mitsui Chemicals America/index2_ocr"
# # # # # #                 ],
# # # # # #                 "Northpoint Commercial Finance": [
# # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_Northpoint Commercial Finance/index1",
# # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_Northpoint Commercial Finance/index2_ocr"
# # # # # #                 ],
# # # # # #                 "iBAS": [
# # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_iBAS/index1",
# # # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_iBAS/index2_ocr"
# # # # # #                 ]
# # # # # #             }
# # # # # #             embeddings = AzureOpenAIEmbeddings(
# # # # # #                 azure_deployment='embeddings-aims',
# # # # # #                 openai_api_version="2024-04-01-preview",
# # # # # #                 azure_endpoint=os.getenv("OPENAI_ENDPOINT_AZURE"),
# # # # # #                 api_key=os.getenv("OPENAI_API_KEY_AZURE")
# # # # # #             )
# # # # # #             faiss_indexes = load_faiss_indexes(account_indexes, embeddings, connection_string)

# # # # # #             # Navigation and Main Content
# # # # # #             styles = {
# # # # # #                 "span": {
# # # # # #                     "border-radius": "0.1rem",
# # # # # #                     "color": "rgb(49, 51, 63)",
# # # # # #                     "margin": "0 0.125rem",
# # # # # #                     "padding": "0.400rem 0.400rem",
# # # # # #                 },
# # # # # #                 "active": {
# # # # # #                     "background-color": "rgba(255, 255, 255, 0.25)",
# # # # # #                 },
# # # # # #             }

# # # # # #             selected = st_navbar(["Home", "Chatbot", "Alerts and Escalation Matrix"],
# # # # # #                                  selected=st.session_state.selected_option, styles=styles)

# # # # # #             # --- APP ---
# # # # # #             if selected == "Home":
# # # # # #                 st.write("# Welcome to AI Support Assistant! 👋")
# # # # # #                 st.markdown(
# # # # # #                 """
# # # # # #                 Welcome to the AI Support Assistant! This tool is designed to streamline your support process by providing quick access to essential information and AI-powered assistance.

# # # # # #                 ### Getting Started:

# # # # # #                 - **☝️ Select Chatbot option from the navigation bar** to begin interacting with the AI Support Assistant.
# # # # # #                 - Make sure to select the correct account in the Alerts and Escalation Matrix tab to view the most relevant information.

# # # # # #                 ### How to Use the App:

# # # # # #                 **1. Chatbot Tab:**  
# # # # # #                 Navigate to the **Chatbot** tab to interact with our AI Assistant. You can ask questions or provide prompts related to your support needs, and the AI will generate detailed responses based on the context provided. This feature is ideal for quickly resolving issues or getting specific information.

# # # # # #                 **Steps:**
# # # # # #                 - Click on the **Chatbot** tab in the navigation bar and select an **Account**.
# # # # # #                 - Type your question or request in the input box at the bottom of the page.
# # # # # #                 - The AI will process your query and provide a response based on the available data.

# # # # # #                 **2. Alerts and Escalation Matrix Tab:**  
# # # # # #                 Visit the **Alerts and Escalation Matrix** tab to view critical alerts and the escalation matrix for specific accounts. This section provides important information about who to contact and the appropriate escalation procedures.

# # # # # #                 **Steps:**
# # # # # #                 - Click on the **Alerts and Escalation Matrix** tab in the navigation bar.
# # # # # #                 - Use the sidebar to select the account you wish to view.
# # # # # #                 - The page will display the relevant alerts and escalation matrix for the selected account.
                
# # # # # #                 If you need assistance at any point, feel free to ask a question in the Chatbot tab.
# # # # # #                 """
# # # # # #                 )
# # # # # #                 st.session_state.selected_option = 'Home'
# # # # # #             elif selected == 'Chatbot':
# # # # # #                 qa_chain = LLMChain(llm=AzureChatOpenAI(
# # # # # #                     model_name='gpt-4o',
# # # # # #                     openai_api_key=os.getenv("OPENAI_API_KEY_AZURE"),
# # # # # #                     azure_endpoint=os.getenv("OPENAI_ENDPOINT_AZURE"),
# # # # # #                     openai_api_version="2024-04-01-preview",
# # # # # #                     temperature=0,
# # # # # #                     max_tokens=4000,
# # # # # #                     streaming=True,
# # # # # #                     verbose=True,
# # # # # #                     model_kwargs={'seed': 123}
# # # # # #                 ), prompt=PromptTemplate.from_template("""
# # # # # #                 ### Instruction ###
# # # # # #                 Given the context below, provide a detailed and accurate answer to the question. Base your response primarily on the provided information. Only include additional context from external sources if absolutely necessary, and clearly identify it as such. DO NOT PARAPHRASE ANYTHING AND GIVE EXACTLY AS ITS GIVEN IN THE DOCUMENT.

# # # # # #                 **Context:**
# # # # # #                 {context}

# # # # # #                 **Question:**
# # # # # #                 {question}

# # # # # #                 ### Guidelines ###
# # # # # #                 1. **Primary Source**: Base your response primarily on the provided context and give the process as exact as given in the document.
# # # # # #                 2. **External Information**: If additional information is needed, clearly label it as "External Information" and keep it to a minimum.
# # # # # #                 3. **Specificity**: Provide detailed and precise information directly related to the query.
# # # # # #                 4. **Separation of Information**: Use headings such as "IT Glue Response" and "External Information" to differentiate the sources.
# # # # # #                 5. **Insufficient Context**: If the provided context does not contain enough information to answer the question, state: "The provided context does not contain enough information to answer the question."
# # # # # #                 6. **Document References**: List the names of all documents accessed, along with a confidence score for each document based on its relevance. 

# # # # # #                 ### Example ###

# # # # # #                 #### IT Glue Response ####
# # # # # #                 [Your answer based on the given context]
                                                       
# # # # # #                 ## External Information ##
# # # # # #                 []

# # # # # #                 #### Alerts and Escalation Matrix ####
# # # # # #                 [Your answer after referring to the specific escalation matrix file for the given account name to determine who should be alerted or to whom the issue should be escalated. Ensure that the appropriate contacts are notified based on the escalation matrix provided in the document.] 
                
# # # # # #                 ### Document Names ###
# # # # # #                 [List of documents and confidence scores (in %) with descending order.] 
                
# # # # # #                 **Answer:**
# # # # # #                 """))
# # # # # #                 run_chatbot(faiss_indexes, blob_service_client, embeddings, qa_chain)
# # # # # #                 st.session_state.selected_option = 'Chatbot'

# # # # # #             elif selected == 'Alerts and Escalation Matrix':
# # # # # #                 with st.sidebar:
# # # # # #                     client_names = ["Select an Account Name"] + account_names
# # # # # #                     selected_client = st.selectbox("**Select Account Name** 🚩", client_names)
# # # # # #                     if selected_client != st.session_state['previous_clientOrg']:
# # # # # #                         st.session_state['clientOrg'] = selected_client
# # # # # #                         st.session_state['previous_clientOrg'] = selected_client

# # # # # #                     if st.session_state['clientOrg'] and st.session_state['clientOrg'] != "Select an Account Name":
# # # # # #                         st.info(f"You are viewing the {st.session_state['clientOrg']} Account")

# # # # # #                 if st.session_state['clientOrg'] and st.session_state['clientOrg'] != "Select an Account Name":
# # # # # #                     alerts_content = load_content_from_blob(blob_service_client, container_name, st.session_state['clientOrg'], "Alerts")
# # # # # #                     escalation_matrix_content = load_content_from_blob(blob_service_client, container_name, st.session_state['clientOrg'], "Escalation Matrix")

# # # # # #                     st.subheader("Alerts")
# # # # # #                     st.markdown(alerts_content)

# # # # # #                     st.subheader("Escalation Matrix")
# # # # # #                     st.markdown(escalation_matrix_content)
# # # # # #                 else:
# # # # # #                     st.subheader("Alerts")
# # # # # #                     st.warning("Please select an account name to view the alerts.")

# # # # # #                     st.subheader("Escalation Matrix")
# # # # # #                     st.warning("Please select an account name to view the escalation matrix.")

# # # # # #             st.sidebar.markdown("""<div style="height: 12vh;"></div>""", unsafe_allow_html=True)
# # # # # #             st.sidebar.markdown(f'## Hello, *{st.session_state["name"]}*')
# # # # # #             if st.sidebar.button("Logout", key="logout_button"):
# # # # # #                 authenticator.logout('Logout', 'sidebar')
# # # # # #                 for key in list(st.session_state.keys()):
# # # # # #                     del st.session_state[key]
# # # # # #                 st.experimental_rerun()

# # # # # # elif st.session_state["authentication_status"] == False:
# # # # # #     st.sidebar.error('Username/password is incorrect')
# # # # # # elif st.session_state["authentication_status"] == None:
# # # # # #     st.sidebar.warning('Please enter your username and password')



# # # # # import os
# # # # # from azure.storage.blob import BlobServiceClient
# # # # # import streamlit as st
# # # # # from yaml.loader import SafeLoader
# # # # # import yaml
# # # # # from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
# # # # # from langchain.chains import LLMChain
# # # # # from langchain.prompts import PromptTemplate
# # # # # from dotenv import load_dotenv
# # # # # import streamlit_authenticator as stauth
# # # # # from streamlit_navigation_bar import st_navbar
# # # # # import logging
# # # # # import pyotp
# # # # # import qrcode
# # # # # import io

# # # # # from aisupport import run_chatbot, initialize_session_state, load_faiss_indexes

# # # # # # Set page configuration
# # # # # st.set_page_config(page_title="AI Support Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="auto")

# # # # # st.markdown(
# # # # #     """
# # # # #     <style>
# # # # #     /* Ensures that the sidebar starts at the top */
# # # # #     .css-1lcbmhc {
# # # # #         padding-top: 0px;
# # # # #     }
# # # # #     /* Adjusts padding around the sidebar's content */
# # # # #     .css-1aumxhk {
# # # # #         padding-top: 0px;
# # # # #     }
# # # # #     </style>
# # # # #     """,
# # # # #     unsafe_allow_html=True
# # # # # )

# # # # # with st.sidebar:
# # # # #     st.image(r"./synoptek.png", width=275)

# # # # # load_dotenv()
# # # # # # Load config
# # # # # connection_string = os.getenv("BLOB_CONNECTION_STRING")
# # # # # container_name = "itgluecopilot"
# # # # # config_blob_name = "config/config.yaml"

# # # # # # BlobServiceClient
# # # # # blob_service_client = BlobServiceClient.from_connection_string(connection_string)
# # # # # container_client = blob_service_client.get_container_client(container_name)

# # # # # # Load the YAML configuration file
# # # # # blob_client = container_client.get_blob_client(config_blob_name)
# # # # # blob_data = blob_client.download_blob().readall()
# # # # # config = yaml.load(io.BytesIO(blob_data), Loader=SafeLoader)

# # # # # authenticator = stauth.Authenticate(
# # # # #     config['credentials'],
# # # # #     config['cookie']['name'],
# # # # #     config['cookie']['key'],
# # # # #     config['cookie']['expiry_days'],
# # # # # )

# # # # # # Configure logging
# # # # # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(level)s - %(message)s", handlers=[logging.StreamHandler()])
# # # # # logger = logging.getLogger(__name__)

# # # # # # Load environment variables
# # # # # load_dotenv()
# # # # # logger.info("Environment variables loaded")

# # # # # # Initialize session state
# # # # # initialize_session_state()

# # # # # # Authentication for App
# # # # # with st.sidebar:
# # # # #     name, authentication_status, username = authenticator.login('Login', 'main')

# # # # # if st.session_state["authentication_status"]:
# # # # #     # Load account names dynamically from the blob
# # # # #     account_names = ["Mitsui Chemicals America", "Northpoint Commercial Finance", "iBAS"]

# # # # #     # Check for OTP Secret and Generate if Not Present
# # # # #     user_data = config['credentials']['usernames'].get(username, {})
# # # # #     otp_secret = user_data.get('otp_secret', "")

# # # # #     if not otp_secret:
# # # # #         otp_secret = pyotp.random_base32()
# # # # #         config['credentials']['usernames'][username]['otp_secret'] = otp_secret
# # # # #         blob_client.upload_blob(yaml.dump(config), overwrite=True)
# # # # #         st.session_state['otp_setup_complete'] = False
# # # # #         st.session_state['show_qr_code'] = True
# # # # #         logger.info("Generated new OTP secret and set show_qr_code to True for user %s", username)
# # # # #     else:
# # # # #         st.session_state['otp_setup_complete'] = True

# # # # #     # Ensure OTP secret is properly handled
# # # # #     if otp_secret:
# # # # #         totp = pyotp.TOTP(otp_secret)
# # # # #         logger.info("Using OTP secret for user %s: %s", username, otp_secret)

# # # # #         if not st.session_state['otp_verified']:
# # # # #             if st.session_state['show_qr_code']:
# # # # #                 st.title("Welcome to AI Support Assistant! 👋")
# # # # #                 otp_uri = totp.provisioning_uri(name=user_data.get('email', ''), issuer_name="AI Support Assistant")
# # # # #                 qr = qrcode.make(otp_uri)
# # # # #                 qr = qr.resize((200, 200))

# # # # #                 st.image(qr, caption="Scan this QR code with your authenticator app")

# # # # #             st.title("AI Support Assistant")
# # # # #             otp_input = st.text_input("Enter the OTP from your authenticator app", type="password")
# # # # #             verify_button_clicked = st.button("Verify OTP")

# # # # #             if verify_button_clicked:
# # # # #                 if totp.verify(otp_input):
# # # # #                     st.session_state['otp_verified'] = True
# # # # #                     st.session_state['show_qr_code'] = False
# # # # #                     blob_client.upload_blob(yaml.dump(config), overwrite=True)
# # # # #                     st.experimental_rerun()
# # # # #                 else:
# # # # #                     st.error("Invalid OTP. Please try again.")
# # # # #         else:
# # # # #             # Load FAISS indexes
# # # # #             account_indexes = {
# # # # #                 "Mitsui Chemicals America": [
# # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_Mitsui Chemicals America/index1",
# # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_Mitsui Chemicals America/index2_ocr"
# # # # #                 ],
# # # # #                 "Northpoint Commercial Finance": [
# # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_Northpoint Commercial Finance/index1",
# # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_Northpoint Commercial Finance/index2_ocr"
# # # # #                 ],
# # # # #                 "iBAS": [
# # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_iBAS/index1",
# # # # #                     "itgluecopilot/Faiss_Index_IT Glue/Index_iBAS/index2_ocr"
# # # # #                 ]
# # # # #             }
# # # # #             embeddings = AzureOpenAIEmbeddings(
# # # # #                 azure_deployment='embeddings-aims',
# # # # #                 openai_api_version="2024-04-01-preview",
# # # # #                 azure_endpoint=os.getenv("OPENAI_ENDPOINT_AZURE"),
# # # # #                 api_key=os.getenv("OPENAI_API_KEY_AZURE")
# # # # #             )
# # # # #             faiss_indexes = load_faiss_indexes(account_indexes, embeddings, connection_string)

# # # # #             # Navigation and Main Content
# # # # #             styles = {
# # # # #                 "span": {
# # # # #                     "border-radius": "0.1rem",
# # # # #                     "color": "rgb(49, 51, 63)",
# # # # #                     "margin": "0 0.125rem",
# # # # #                     "padding": "0.400rem 0.400rem",
# # # # #                     "color": "orange",
# # # # #                     "text-decoration": "underline",
# # # # #                 },
# # # # #                 "active": {
# # # # #                     "background-color": "rgba(255, 255, 255, 0.25)"
# # # # #                     # "color": "orange"
# # # # #                 },
# # # # #             }

# # # # #             selected = st_navbar(["Home", "Chatbot", "Alerts and Escalation Matrix"],
# # # # #                                  selected=st.session_state.selected_option, styles=styles)

# # # # #             # --- APP ---
# # # # #             if selected == "Home":
# # # # #                 st.write("# Welcome to AI Support Assistant! 👋")
# # # # #                 st.markdown(
# # # # #                 """
# # # # #                 Welcome to the AI Support Assistant! This tool is designed to streamline your support process by providing quick access to essential information and AI-powered assistance.

# # # # #                 ### Getting Started:

# # # # #                 - **☝️ Select Chatbot option from the navigation bar** to begin interacting with the AI Support Assistant.
# # # # #                 - Make sure to select the correct account in the Alerts and Escalation Matrix tab to view the most relevant information.

# # # # #                 ### How to Use the App:

# # # # #                 **1. Chatbot Tab:**  
# # # # #                 Navigate to the **Chatbot** tab to interact with our AI Assistant. You can ask questions or provide prompts related to your support needs, and the AI will generate detailed responses based on the context provided. This feature is ideal for quickly resolving issues or getting specific information.

# # # # #                 **Steps:**
# # # # #                 - Click on the **Chatbot** tab in the navigation bar and select an **Account**.
# # # # #                 - Type your question or request in the input box at the bottom of the page.
# # # # #                 - The AI will process your query and provide a response based on the available data.

# # # # #                 **2. Alerts and Escalation Matrix Tab:**  
# # # # #                 Visit the **Alerts and Escalation Matrix** tab to view critical alerts and the escalation matrix for specific accounts. This section provides important information about who to contact and the appropriate escalation procedures.

# # # # #                 **Steps:**
# # # # #                 - Click on the **Alerts and Escalation Matrix** tab in the navigation bar.
# # # # #                 - Use the sidebar to select the account you wish to view.
# # # # #                 - The page will display the relevant alerts and escalation matrix for the selected account.
                
# # # # #                 If you need assistance at any point, feel free to ask a question in the Chatbot tab.
# # # # #                 """
# # # # #                 )
# # # # #                 st.session_state.selected_option = 'Home'
# # # # #             elif selected == 'Chatbot':
# # # # #                 qa_chain = LLMChain(llm=AzureChatOpenAI(
# # # # #                     model_name='gpt-4o',
# # # # #                     openai_api_key=os.getenv("OPENAI_API_KEY_AZURE"),
# # # # #                     azure_endpoint=os.getenv("OPENAI_ENDPOINT_AZURE"),
# # # # #                     openai_api_version="2024-04-01-preview",
# # # # #                     temperature=0,
# # # # #                     max_tokens=4000,
# # # # #                     streaming=True,
# # # # #                     verbose=True,
# # # # #                     model_kwargs={'seed': 123}
# # # # #                 ), prompt=PromptTemplate.from_template("""
# # # # #                 ### Instruction ###
# # # # #                 Given the context below, provide a detailed and accurate answer to the question. Base your response primarily on the provided information. Only include additional context from external sources if absolutely necessary, and clearly identify it as such. DO NOT PARAPHRASE ANYTHING AND GIVE EXACTLY AS ITS GIVEN IN THE DOCUMENT.

# # # # #                 **Context:**
# # # # #                 {context}

# # # # #                 **Question:**
# # # # #                 {question}

# # # # #                 ### Guidelines ###
# # # # #                 1. **Primary Source**: Base your response primarily on the provided context and give the process as exact as given in the document.
# # # # #                 2. **External Information**: If additional information is needed, clearly label it as "External Information" and keep it to a minimum.
# # # # #                 3. **Specificity**: Provide detailed and precise information directly related to the query.
# # # # #                 4. **Separation of Information**: Use headings such as "IT Glue Response" and "External Information" to differentiate the sources.
# # # # #                 5. **Insufficient Context**: If the provided context does not contain enough information to answer the question, state: "The provided context does not contain enough information to answer the question."
# # # # #                 6. **Document References**: List the names of all documents accessed, along with a confidence score for each document based on its relevance. 

# # # # #                 ### Example ###

# # # # #                 #### IT Glue Response ####
# # # # #                 [Your answer based on the given context]
                                                       
# # # # #                 ## External Information ##
# # # # #                 []

# # # # #                 #### Alerts and Escalation Matrix ####
# # # # #                 [Your answer after referring to the specific escalation matrix file for the given account name to determine who should be alerted or to whom the issue should be escalated. Ensure that the appropriate contacts are notified based on the escalation matrix provided in the document.] 
                
# # # # #                 ### Document Names ###
# # # # #                 [List of documents and confidence scores (in %) with descending order.] 
                
# # # # #                 **Answer:**
# # # # #                 """))
# # # # #                 run_chatbot(faiss_indexes, blob_service_client, embeddings, qa_chain)
# # # # #                 st.session_state.selected_option = 'Chatbot'

# # # # #             elif selected == 'Alerts and Escalation Matrix':
# # # # #                 with st.sidebar:
# # # # #                     client_names = ["Select an Account Name"] + account_names
# # # # #                     selected_client = st.selectbox("**Select Account Name** 🚩", client_names)
# # # # #                     if selected_client != st.session_state['previous_clientOrg']:
# # # # #                         st.session_state['clientOrg'] = selected_client
# # # # #                         st.session_state['previous_clientOrg'] = selected_client

# # # # #                     if st.session_state['clientOrg'] and st.session_state['clientOrg'] != "Select an Account Name":
# # # # #                         st.info(f"You are viewing the {st.session_state['clientOrg']} Account")

# # # # #                 if st.session_state['clientOrg'] and st.session_state['clientOrg'] != "Select an Account Name":
# # # # #                     alerts_content = load_content_from_blob(blob_service_client, container_name, st.session_state['clientOrg'], "Alerts")
# # # # #                     escalation_matrix_content = load_content_from_blob(blob_service_client, container_name, st.session_state['clientOrg'], "Escalation Matrix")

# # # # #                     st.subheader("Alerts")
# # # # #                     st.markdown(alerts_content)

# # # # #                     st.subheader("Escalation Matrix")
# # # # #                     st.markdown(escalation_matrix_content)
# # # # #                 else:
# # # # #                     st.subheader("Alerts")
# # # # #                     st.warning("Please select an account name to view the alerts.")

# # # # #                     st.subheader("Escalation Matrix")
# # # # #                     st.warning("Please select an account name to view the escalation matrix.")

# # # # #             st.sidebar.markdown("""<div style="height: 12vh;"></div>""", unsafe_allow_html=True)
# # # # #             st.sidebar.markdown(f'## Hello, *{st.session_state["name"]}*')
# # # # #             if st.sidebar.button("Logout", key="logout_button"):
# # # # #                 authenticator.logout('Logout', 'sidebar')
# # # # #                 for key in list(st.session_state.keys()):
# # # # #                     del st.session_state[key]
# # # # #                 st.experimental_rerun()

# # # # # elif st.session_state["authentication_status"] == False:
# # # # #     st.sidebar.error('Username/password is incorrect')
# # # # #     st.write("# Welcome to AI Support Assistant! 👋")
# # # # #     st.markdown(
# # # # #         """
# # # # #         Please enter your username and password to log in.
# # # # #         """
# # # # #     )
# # # # # elif st.session_state["authentication_status"] == None:
# # # # #     st.sidebar.warning('Please enter your username and password')
# # # # #     st.write("# Welcome to AI Support Assistant! 👋")
# # # # #     st.markdown(
# # # # #         """
# # # # #         Please enter your username and password to log in.
# # # # #         """
# # # # #     )


# # import os
# # from azure.storage.blob import BlobServiceClient
# # import streamlit as st
# # from yaml.loader import SafeLoader
# # import yaml
# # from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
# # from langchain.chains import LLMChain
# # from langchain.prompts import PromptTemplate
# # from dotenv import load_dotenv
# # import streamlit_authenticator as stauth
# # from streamlit_navigation_bar import st_navbar
# # import logging
# # import pyotp
# # import qrcode
# # import io

# # from aisupport import run_chatbot, initialize_session_state, load_faiss_indexes

# # # Set page configuration
# # st.set_page_config(page_title="AI Support Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="auto")

# # # st.markdown(
# # #     """
# # #     <style>
# # #     /* Ensures that the sidebar starts at the top */
# # #     .css-1lcbmhc {
# # #         padding-top: 0px;
# # #     }
# # #     /* Adjusts padding around the sidebar's content */
# # #     .css-1aumxhk {
# # #         padding-top: 0px;
# # #     }
# # #     </style>
# # #     """,
# # #     unsafe_allow_html=True
# # # )


# # with st.sidebar:
# #     st.image(r"./synoptek.png", width=275)

# # load_dotenv()
# # # Load config
# # connection_string = os.getenv("BLOB_CONNECTION_STRING")
# # container_name = "itgluecopilot"
# # config_blob_name = "config/config.yaml"

# # # BlobServiceClient
# # blob_service_client = BlobServiceClient.from_connection_string(connection_string)
# # container_client = blob_service_client.get_container_client(container_name)

# # # Load the YAML configuration file
# # blob_client = container_client.get_blob_client(config_blob_name)
# # blob_data = blob_client.download_blob().readall()
# # config = yaml.load(io.BytesIO(blob_data), Loader=SafeLoader)

# # authenticator = stauth.Authenticate(
# #     config['credentials'],
# #     config['cookie']['name'],
# #     config['cookie']['key'],
# #     config['cookie']['expiry_days'],
# # )


# # # Configure logging
# # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
# # logger = logging.getLogger(__name__)


# # # logger.info("Environment variables loaded")

# # # Load environment variables
# # load_dotenv()
# # logger.info("Environment variables loaded")

# # # Initialize session state
# # initialize_session_state()

# # # Authentication for App
# # with st.sidebar:
# #     name, authentication_status, username = authenticator.login('Login', 'main')

# # def load_content_from_blob(blob_service_client, container_name, account_name, content_type):
# #     """Load specific content from a .txt file in Azure Blob Storage based on the account name and content type."""
# #     blob_name = f"Documents/{account_name}/{content_type}.txt"

# #     try:
# #         blob_client = blob_service_client.get_blob_client(container_name, blob_name)
# #         blob_data = blob_client.download_blob().readall().decode('utf-8')
# #         return blob_data
# #     except Exception as e:
# #         # logger.error(f"Error loading {content_type} for {account_name}: {e}")
# #         return f"{content_type} content not found for {account_name}. Please ensure the file exists."

# # if st.session_state["authentication_status"]:
# #     # Load account names dynamically from the blob
# #     account_names = ["Mitsui Chemicals America", "Northpoint Commercial Finance", "iBAS"]

# #     # Check for OTP Secret and Generate if Not Present
# #     user_data = config['credentials']['usernames'].get(username, {})
# #     otp_secret = user_data.get('otp_secret', "")

# #     if not otp_secret:
# #         otp_secret = pyotp.random_base32()
# #         config['credentials']['usernames'][username]['otp_secret'] = otp_secret
# #         blob_client.upload_blob(yaml.dump(config), overwrite=True)
# #         st.session_state['otp_setup_complete'] = False
# #         st.session_state['show_qr_code'] = True
# #         logger.info("Generated new OTP secret and set show_qr_code to True for user %s", username)
# #     else:
# #         st.session_state['otp_setup_complete'] = True

# #     # Ensure OTP secret is properly handled
# #     if otp_secret:
# #         totp = pyotp.TOTP(otp_secret)
# #         logger.info("Using OTP secret for user %s: %s", username, otp_secret)

# #         if not st.session_state['otp_verified']:
# #             if st.session_state['show_qr_code']:
# #                 st.title("Welcome to AI Support Assistant! 👋")
# #                 otp_uri = totp.provisioning_uri(name=user_data.get('email', ''), issuer_name="AI Support Assistant")
# #                 qr = qrcode.make(otp_uri)
# #                 qr = qr.resize((200, 200))

# #                 st.image(qr, caption="Scan this QR code with your authenticator app")

# #             st.title("AI Support Assistant")
# #             otp_input = st.text_input("Enter the OTP from your authenticator app", type="password")
# #             verify_button_clicked = st.button("Verify OTP")

# #             if verify_button_clicked:
# #                 if totp.verify(otp_input):
# #                     st.session_state['otp_verified'] = True
# #                     st.session_state['show_qr_code'] = False
# #                     blob_client.upload_blob(yaml.dump(config), overwrite=True)
# #                     st.experimental_rerun()
# #                 else:
# #                     st.error("Invalid OTP. Please try again.")
# #         else:
# #             # Load FAISS indexes
# #             account_indexes = {
# #                 "Mitsui Chemicals America": [
# #                     r"./Faiss_Index_IT Glue/Index_Mitsui Chemicals America/index1",
# #                     r"./Faiss_Index_IT Glue/Index_Mitsui Chemicals America/index2_ocr"
# #                 ],
# #                 "Northpoint Commercial Finance": [
# #                     r"./Faiss_Index_IT Glue/Index_Northpoint Commercial Finance/index1",
# #                     r"./Faiss_Index_IT Glue/Index_Northpoint Commercial Finance/index2_ocr"
# #                 ],
# #                 "iBAS": [
# #                     r"./Faiss_Index_IT Glue/Index_iBAS/index1",
# #                     r"./Faiss_Index_IT Glue/Index_iBAS/index2_ocr"
# #                 ]
# #             }
# #             embeddings = AzureOpenAIEmbeddings(
# #                 azure_deployment='embeddings-aims',
# #                 openai_api_version="2024-04-01-preview",
# #                 azure_endpoint=os.getenv("OPENAI_ENDPOINT_AZURE"),
# #                 api_key=os.getenv("OPENAI_API_KEY_AZURE")
# #             )
# #             faiss_indexes = load_faiss_indexes(account_indexes, embeddings)#, connection_string)

# #             # Navigation and Main Content
# #             styles = {
# #                 "span": {
# #                     "border-radius": "0.1rem",
# #                     "color": "orange",
# #                     "margin": "0 0.125rem",
# #                     "padding": "0.400rem 0.400rem",
# #                 },
# #                 "active": {
# #                     "background-color": "rgba(255, 255, 255, 0.25)",
# #                     "color": "orange",  # Active text color to orange
# #                     "text-decoration": "underline",  # Underline the active text
# #                 },
# #             }

# #             # Render the navigation bar and update session state
# #             selected = st_navbar(["Home", "Chatbot", "Alerts and Escalation Matrix"],
# #                                  selected=st.session_state.get('selected_option', 'Home'), styles=styles)
            
            
# #             # Update the session state with the selected option
# #             st.session_state.selected_option = selected

# #             # --- APP ---
# #             if selected == "Home":
# #                 st.write("# Welcome to AI Support Assistant! 👋")
# #                 st.markdown(
# #                 """
# #                 Welcome to the AI Support Assistant! This tool is designed to streamline your support process by providing quick access to essential information and AI-powered assistance.

# #                 ### Getting Started:

# #                 - **☝️ Select Chatbot option from the navigation bar** to begin interacting with the AI Support Assistant.
# #                 - Make sure to select the correct account in the Alerts and Escalation Matrix tab to view the most relevant information.

# #                 ### How to Use the App:

# #                 **1. Chatbot Tab:**  
# #                 Navigate to the **Chatbot** tab to interact with our AI Assistant. You can ask questions or provide prompts related to your support needs, and the AI will generate detailed responses based on the context provided. This feature is ideal for quickly resolving issues or getting specific information.

# #                 **Steps:**
# #                 - Click on the **Chatbot** tab in the navigation bar and select an **Account**.
# #                 - Type your question or request in the input box at the bottom of the page.
# #                 - The AI will process your query and provide a response based on the available data.

# #                 **2. Alerts and Escalation Matrix Tab:**  
# #                 Visit the **Alerts and Escalation Matrix** tab to view critical alerts and the escalation matrix for specific accounts. This section provides important information about who to contact and the appropriate escalation procedures.

# #                 **Steps:**
# #                 - Click on the **Alerts and Escalation Matrix** tab in the navigation bar.
# #                 - Use the sidebar to select the account you wish to view.
# #                 - The page will display the relevant alerts and escalation matrix for the selected account.
                
# #                 If you need assistance at any point, feel free to ask a question in the Chatbot tab.
# #                 """
# #                 )
# #             elif selected == 'Chatbot':
# #                 qa_chain = LLMChain(llm=AzureChatOpenAI(
# #                     model_name='gpt-4o',
# #                     openai_api_key=os.getenv("OPENAI_API_KEY_AZURE"),
# #                     azure_endpoint=os.getenv("OPENAI_ENDPOINT_AZURE"),
# #                     openai_api_version="2024-04-01-preview",
# #                     temperature=0,
# #                     max_tokens=4000,
# #                     streaming=True,
# #                     verbose=True,
# #                     model_kwargs={'seed': 123}
# #                 ), prompt=PromptTemplate.from_template("""
# #                 ### Instruction ###
# #                 Given the context below, provide a detailed and accurate answer to the question. Base your response primarily on the provided information. Only include additional context from external sources if absolutely necessary, and clearly identify it as such. DO NOT PARAPHRASE ANYTHING AND GIVE EXACTLY AS ITS GIVEN IN THE DOCUMENT.

# #                 **Context:**
# #                 {context}

# #                 **Question:**
# #                 {question}

# #                 ### Guidelines ###
# #                 1. **Primary Source**: Base your response primarily on the provided context and give the process as exact as given in the document.
# #                 2. **External Information**: If additional information is needed, clearly label it as "External Information" and keep it to a minimum.
# #                 3. **Specificity**: Provide detailed and precise information directly related to the query.
# #                 4. **Separation of Information**: Use headings such as "IT Glue Response" and "External Information" to differentiate the sources.
# #                 5. **Insufficient Context**: If the provided context does not contain enough information to answer the question, state: "The provided context does not contain enough information to answer the question."
# #                 6. **Document References**: List the names of all documents accessed, along with a confidence score for each document based on its relevance. 

# #                 ### Example ###

# #                 #### IT Glue Response ####
# #                 [Your answer based on the given context]
                                                       
# #                 ## External Information ##
# #                 []

# #                 #### Alerts and Escalation Matrix ####
# #                 [Your answer after referring to the specific escalation matrix file for the given account name to determine who should be alerted or to whom the issue should be escalated. Ensure that the appropriate contacts are notified based on the escalation matrix provided in the document.] 
                
# #                 ### Document Names ###
# #                 [List of documents and confidence scores (in %) with descending order.] 
                
# #                 **Answer:**
# #                 """))
# #                 run_chatbot(faiss_indexes, blob_service_client, embeddings, qa_chain)
            
# #             elif selected == 'Alerts and Escalation Matrix':
# #                 with st.sidebar:
# #                     client_names = ["Select an Account Name"] + account_names
# #                     selected_client = st.selectbox("**Select Account Name** 🚩", client_names)
# #                     if selected_client != st.session_state['previous_clientOrg']:
# #                         st.session_state['clientOrg'] = selected_client
# #                         st.session_state['previous_clientOrg'] = selected_client

# #                     if st.session_state['clientOrg'] and st.session_state['clientOrg'] != "Select an Account Name":
# #                         st.info(f"You are viewing the {st.session_state['clientOrg']} Account")

# #                 if st.session_state['clientOrg'] and st.session_state['clientOrg'] != "Select an Account Name":
# #                     alerts_content = load_content_from_blob(blob_service_client, container_name, st.session_state['clientOrg'], "Alerts")
# #                     escalation_matrix_content = load_content_from_blob(blob_service_client, container_name, st.session_state['clientOrg'], "Escalation Matrix")

# #                     st.subheader("Alerts")
# #                     st.markdown(
# #                         f"""
# #                         <div style="
# #                             border: 2px solid #ffcc00; 
# #                             padding: 15px; 
# #                             border-radius: 10px; 
# #                             background-color: #fff7e6;">
# #                             {alerts_content}
# #                         </div>
# #                         """,
# #                         unsafe_allow_html=True
# #                     )

# #                     st.subheader("Escalation Matrix")
# #                     st.markdown(
# #                         f"""
# #                         <div style="
# #                             border: 2px solid #0066cc; 
# #                             padding: 15px; 
# #                             border-radius: 10px; 
# #                             background-color: #e6f2ff;">
# #                             {escalation_matrix_content}
# #                         </div>
# #                         """,
# #                         unsafe_allow_html=True
# #                     )
# #                 else:
# #                     st.subheader("Alerts")
# #                     st.warning("Please select an account name to view the alerts.")

# #                     st.subheader("Escalation Matrix")
# #                     st.warning("Please select an account name to view the escalation matrix.")

# #             st.sidebar.markdown("""<div style="height: 12vh;"></div>""", unsafe_allow_html=True)
# #             st.sidebar.markdown(f'## Hello, *{st.session_state["name"]}*')
# #             if st.sidebar.button("Logout", key="logout_button"):
# #                 authenticator.logout('Logout', 'sidebar')
# #                 for key in list(st.session_state.keys()):
# #                     del st.session_state[key]
# #                 st.experimental_rerun()

# # elif st.session_state["authentication_status"] == False:
# #     st.sidebar.error('Username/password is incorrect')
# #     st.write("# Welcome to AI Support Assistant! 👋")
# #     st.markdown(
# #         """
# #         Please enter your username and password to log in.
# #         """
# #     )
# # elif st.session_state["authentication_status"] == None:
# #     st.sidebar.warning('Please enter your username and password')
# #     st.write("# Welcome to AI Support Assistant! 👋")
# #     st.markdown(
# #         """
# #         Please enter your username and password to log in.
# #         """
# #     )


# import os
# from azure.storage.blob import BlobServiceClient
# import streamlit as st
# from yaml.loader import SafeLoader
# import yaml
# from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import streamlit_authenticator as stauth
# from streamlit_navigation_bar import st_navbar
# import logging
# import pyotp
# import qrcode
# import io

# from aisupport import run_chatbot, initialize_session_state, load_faiss_indexes

# # Set page configuration
# st.set_page_config(page_title="AI Support Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="auto")

# # Navbar at the top
# styles = {
#     "span": {
#         "border-radius": "0.1rem",
#         "color": "orange",
#         "margin": "0 0.125rem",
#         "padding": "0.400rem 0.400rem",
#     },
#     "active": {
#         "background-color": "rgba(255, 255, 255, 0.25)",
#         "color": "orange",  # Active text color to orange
#         "text-decoration": "underline",  # Underline the active text
#     },
# }

# selected = st_navbar(["Home", "Chatbot", "Alerts and Escalation Matrix"], 
#                      selected=st.session_state.get('selected_option', 'Home'), 
#                      styles=styles)

# # Initialize session state
# initialize_session_state()

# # Load environment variables
# load_dotenv()

# # Load config from Azure Blob Storage
# connection_string = os.getenv("BLOB_CONNECTION_STRING")
# container_name = "itgluecopilot"
# config_blob_name = "config/config.yaml"

# # BlobServiceClient
# blob_service_client = BlobServiceClient.from_connection_string(connection_string)
# container_client = blob_service_client.get_container_client(container_name)

# # Load the YAML configuration file
# blob_client = container_client.get_blob_client(config_blob_name)
# blob_data = blob_client.download_blob().readall()
# config = yaml.load(io.BytesIO(blob_data), Loader=SafeLoader)

# authenticator = stauth.Authenticate(
#     config['credentials'],
#     config['cookie']['name'],
#     config['cookie']['key'],
#     config['cookie']['expiry_days'],
# )

# # Configure logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
# logger = logging.getLogger(__name__)

# # logger.info("Environment variables loaded")
# logger.info("Environment variables loaded")


# # Update session state with the selected option
# st.session_state.selected_option = selected

# with st.sidebar:
#     st.image(r"./synoptek.png", width=275)

# # Authentication for App
# with st.sidebar:
#     name, authentication_status, username = authenticator.login('Login', 'main')

# def load_content_from_blob(blob_service_client, container_name, account_name, content_type):
#     """Load specific content from a .txt file in Azure Blob Storage based on the account name and content type."""
#     blob_name = f"Documents/{account_name}/{content_type}.txt"

#     try:
#         blob_client = blob_service_client.get_blob_client(container_name, blob_name)
#         blob_data = blob_client.download_blob().readall().decode('utf-8')
#         return blob_data
#     except Exception as e:
#         # logger.error(f"Error loading {content_type} for {account_name}: {e}")
#         return f"{content_type} content not found for {account_name}. Please ensure the file exists."

# if st.session_state["authentication_status"]:
#     # Load account names dynamically from the blob
#     account_names = ["Mitsui Chemicals America", "Northpoint Commercial Finance", "iBAS"]

#     # Check for OTP Secret and Generate if Not Present
#     user_data = config['credentials']['usernames'].get(username, {})
#     otp_secret = user_data.get('otp_secret', "")

#     if not otp_secret:
#         otp_secret = pyotp.random_base32()
#         config['credentials']['usernames'][username]['otp_secret'] = otp_secret
#         blob_client.upload_blob(yaml.dump(config), overwrite=True)
#         st.session_state['otp_setup_complete'] = False
#         st.session_state['show_qr_code'] = True
#         logger.info("Generated new OTP secret and set show_qr_code to True for user %s", username)
#     else:
#         st.session_state['otp_setup_complete'] = True

#     # Ensure OTP secret is properly handled
#     if otp_secret:
#         totp = pyotp.TOTP(otp_secret)
#         logger.info("Using OTP secret for user %s: %s", username, otp_secret)

#         if not st.session_state['otp_verified']:
#             if st.session_state['show_qr_code']:
#                 st.title("Welcome to AI Support Assistant! 👋")
#                 otp_uri = totp.provisioning_uri(name=user_data.get('email', ''), issuer_name="AI Support Assistant")
#                 qr = qrcode.make(otp_uri)
#                 qr = qr.resize((200, 200))

#                 st.image(qr, caption="Scan this QR code with your authenticator app")

#             st.title("AI Support Assistant")
#             otp_input = st.text_input("Enter the OTP from your authenticator app", type="password")
#             verify_button_clicked = st.button("Verify OTP")

#             if verify_button_clicked:
#                 if totp.verify(otp_input):
#                     st.session_state['otp_verified'] = True
#                     st.session_state['show_qr_code'] = False
#                     blob_client.upload_blob(yaml.dump(config), overwrite=True)
#                     st.experimental_rerun()
#                 else:
#                     st.error("Invalid OTP. Please try again.")
#         else:
#             # Load FAISS indexes
#             account_indexes = {
#                 "Mitsui Chemicals America": [
#                     r"./Faiss_Index_IT Glue/Index_Mitsui Chemicals America/index1",
#                     r"./Faiss_Index_IT Glue/Index_Mitsui Chemicals America/index2_ocr"
#                 ],
#                 "Northpoint Commercial Finance": [
#                     r"./Faiss_Index_IT Glue/Index_Northpoint Commercial Finance/index1",
#                     r"./Faiss_Index_IT Glue/Index_Northpoint Commercial Finance/index2_ocr"
#                 ],
#                 "iBAS": [
#                     r"./Faiss_Index_IT Glue/Index_iBAS/index1",
#                     r"./Faiss_Index_IT Glue/Index_iBAS/index2_ocr"
#                 ]
#             }
#             embeddings = AzureOpenAIEmbeddings(
#                 azure_deployment='embeddings-aims',
#                 openai_api_version="2024-04-01-preview",
#                 azure_endpoint=os.getenv("OPENAI_ENDPOINT_AZURE"),
#                 api_key=os.getenv("OPENAI_API_KEY_AZURE")
#             )
#             faiss_indexes = load_faiss_indexes(account_indexes, embeddings)

#             # --- APP ---
#             if selected == "Home":
#                 st.write("# Welcome to AI Support Assistant! 👋")
#                 st.markdown(
#                 """
#                 Welcome to the AI Support Assistant! This tool is designed to streamline your support process by providing quick access to essential information and AI-powered assistance.

#                 ### Getting Started:

#                 - **☝️ Select Chatbot option from the navigation bar** to begin interacting with the AI Support Assistant.
#                 - Make sure to select the correct account in the Alerts and Escalation Matrix tab to view the most relevant information.

#                 ### How to Use the App:

#                 **1. Chatbot Tab:**  
#                 Navigate to the **Chatbot** tab to interact with our AI Assistant. You can ask questions or provide prompts related to your support needs, and the AI will generate detailed responses based on the context provided. This feature is ideal for quickly resolving issues or getting specific information.

#                 **Steps:**
#                 - Click on the **Chatbot** tab in the navigation bar and select an **Account**.
#                 - Type your question or request in the input box at the bottom of the page.
#                 - The AI will process your query and provide a response based on the available data.

#                 **2. Alerts and Escalation Matrix Tab:**  
#                 Visit the **Alerts and Escalation Matrix** tab to view critical alerts and the escalation matrix for specific accounts. This section provides important information about who to contact and the appropriate escalation procedures.

#                 **Steps:**
#                 - Click on the **Alerts and Escalation Matrix** tab in the navigation bar.
#                 - Use the sidebar to select the account you wish to view.
#                 - The page will display the relevant alerts and escalation matrix for the selected account.
                
#                 If you need assistance at any point, feel free to ask a question in the Chatbot tab.
#                 """
#                 )
#             elif selected == 'Chatbot':
#                 qa_chain = LLMChain(llm=AzureChatOpenAI(
#                     model_name='gpt-4o',
#                     openai_api_key=os.getenv("OPENAI_API_KEY_AZURE"),
#                     azure_endpoint=os.getenv("OPENAI_ENDPOINT_AZURE"),
#                     openai_api_version="2024-04-01-preview",
#                     temperature=0,
#                     max_tokens=4000,
#                     streaming=True,
#                     verbose=True,
#                     model_kwargs={'seed': 123}
#                 ), prompt=PromptTemplate.from_template("""
#                 ### Instruction ###
#                 Given the context below, provide a detailed and accurate answer to the question. Base your response primarily on the provided information. Only include additional context from external sources if absolutely necessary, and clearly identify it as such. DO NOT PARAPHRASE ANYTHING AND GIVE EXACTLY AS ITS GIVEN IN THE DOCUMENT.

#                 **Context:**
#                 {context}

#                 **Question:**
#                 {question}

#                 ### Guidelines ###
#                 1. **Primary Source**: Base your response primarily on the provided context and give the process as exact as given in the document.
#                 2. **External Information**: If additional information is needed, clearly label it as "External Information" and keep it to a minimum.
#                 3. **Specificity**: Provide detailed and precise information directly related to the query.
#                 4. **Separation of Information**: Use headings such as "IT Glue Response" and "External Information" to differentiate the sources.
#                 5. **Insufficient Context**: If the provided context does not contain enough information to answer the question, state: "The provided context does not contain enough information to answer the question."
#                 6. **Document References**: List the names of all documents accessed, along with a confidence score for each document based on its relevance. 

#                 ### Example ###

#                 #### IT Glue Response ####
#                 [Your answer based on the given context]
                                                       
#                 ## External Information ##
#                 []

#                 #### Alerts and Escalation Matrix ####
#                 [Your answer after referring to the specific escalation matrix file for the given account name to determine who should be alerted or to whom the issue should be escalated. Ensure that the appropriate contacts are notified based on the escalation matrix provided in the document.] 
                
#                 ### Document Names ###
#                 [List of documents and confidence scores (in %) with descending order.] 
                
#                 **Answer:**
#                 """))
#                 run_chatbot(faiss_indexes, blob_service_client, embeddings, qa_chain)
            
#             elif selected == 'Alerts and Escalation Matrix':
#                 with st.sidebar:
#                     client_names = ["Select an Account Name"] + account_names
#                     selected_client = st.selectbox("**Select Account Name** 🚩", client_names)
#                     if selected_client != st.session_state['previous_clientOrg']:
#                         st.session_state['clientOrg'] = selected_client
#                         st.session_state['previous_clientOrg'] = selected_client

#                     if st.session_state['clientOrg'] and st.session_state['clientOrg'] != "Select an Account Name":
#                         st.info(f"You are viewing the {st.session_state['clientOrg']} Account")

#                 if st.session_state['clientOrg'] and st.session_state['clientOrg'] != "Select an Account Name":
#                     alerts_content = load_content_from_blob(blob_service_client, container_name, st.session_state['clientOrg'], "Alerts")
#                     escalation_matrix_content = load_content_from_blob(blob_service_client, container_name, st.session_state['clientOrg'], "Escalation Matrix")

#                     st.subheader("Alerts")
#                     st.markdown(
#                         f"""
#                         <div style="
#                             border: 2px solid #ffcc00; 
#                             padding: 15px; 
#                             border-radius: 10px; 
#                             background-color: #fff7e6;">
#                             {alerts_content}
#                         </div>
#                         """,
#                         unsafe_allow_html=True
#                     )

#                     st.subheader("Escalation Matrix")
#                     st.markdown(
#                         f"""
#                         <div style="
#                             border: 2px solid #0066cc; 
#                             padding: 15px; 
#                             border-radius: 10px; 
#                             background-color: #e6f2ff;">
#                             {escalation_matrix_content}
#                         </div>
#                         """,
#                         unsafe_allow_html=True
#                     )
#                 else:
#                     st.subheader("Alerts")
#                     st.warning("Please select an account name to view the alerts.")

#                     st.subheader("Escalation Matrix")
#                     st.warning("Please select an account name to view the escalation matrix.")

#             st.sidebar.markdown("""<div style="height: 12vh;"></div>""", unsafe_allow_html=True)
#             st.sidebar.markdown(f'## Hello, *{st.session_state["name"]}*')
#             if st.sidebar.button("Logout", key="logout_button"):
#                 authenticator.logout('Logout', 'sidebar')
#                 for key in list(st.session_state.keys()):
#                     del st.session_state[key]
#                 st.experimental_rerun()

# elif st.session_state["authentication_status"] == False:
#     st.sidebar.error('Username/password is incorrect')
#     st.write("# Welcome to AI Support Assistant! 👋")
#     st.markdown(
#         """
#         Please enter your username and password to log in.
#         """
#     )
# elif st.session_state["authentication_status"] == None:
#     st.sidebar.warning('Please enter your username and password')
#     st.write("# Welcome to AI Support Assistant! 👋")
#     st.markdown(
#         """
#         Please enter your username and password to log in.
#         """
#     )


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

from chatbot import run_chatbot, initialize_session_state, load_faiss_indexes

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
config_blob_name = "config/config.yaml"

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

# logger.info("Environment variables loaded")
logger.info("Environment variables loaded")


# Update session state with the selected option
st.session_state.selected_option = selected

with st.sidebar:
    st.image(r"./synoptek.png", width=275)

# Authentication for App
with st.sidebar:
    name, authentication_status, username = authenticator.login('Login', 'main')

def load_content_from_blob(blob_service_client, container_name, account_name, content_type):
    """Load specific content from a .txt file in Azure Blob Storage based on the account name and content type."""
    blob_name = f"Documents/{account_name}_{content_type}.txt"

    try:
        blob_client = blob_service_client.get_blob_client(container_name, blob_name)
        blob_data = blob_client.download_blob().readall().decode('utf-8')
        return blob_data
    except Exception as e:
        logger.error(f"Error loading {content_type} for {account_name}: {e}")
        return f"{content_type} content not found for {account_name}. Please ensure the file exists."

if st.session_state["authentication_status"]:
    # Load account names dynamically from the blob
    account_names = ["Mitsui Chemicals America", "Northpoint Commercial Finance", "iBAS"]

    # Check for OTP Secret and Generate if Not Present
    user_data = config['credentials']['usernames'].get(username, {})
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

            st.title("AI Support Assistant")
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
                ]
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

                - **☝️ Select Chatbot option from the navigation bar** to begin interacting with the AI Support Assistant.
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
                
                If you need assistance at any point, feel free to ask a question in the Chatbot tab.
                """
                )
            elif selected == 'Chatbot':
                qa_chain = LLMChain(llm=AzureChatOpenAI(
                    model_name='gpt-4o',
                    openai_api_key=os.getenv("OPENAI_API_KEY_AZURE"),
                    azure_endpoint=os.getenv("OPENAI_ENDPOINT_AZURE"),
                    openai_api_version="2024-04-01-preview",
                    temperature=0,
                    max_tokens=4000,
                    streaming=True,
                    verbose=True,
                    model_kwargs={'seed': 123}
                ), prompt=PromptTemplate.from_template("""
                ### Instruction ###
                Given the context below, provide a detailed and accurate answer to the question. Base your response primarily on the provided information. Only include additional context from external sources if absolutely necessary, and clearly identify it as such. DO NOT PARAPHRASE ANYTHING AND GIVE EXACTLY AS ITS GIVEN IN THE DOCUMENT.

                **Context:**
                {context}

                **Question:**
                {question}

                ### Guidelines ###
                1. **Primary Source**: Base your response primarily on the provided context and give the process as exact as given in the document.
                2. **External Information**: If additional information is needed, clearly label it as "External Information" and keep it to a minimum.
                3. **Specificity**: Provide detailed and precise information directly related to the query.
                4. **Separation of Information**: Use headings such as "IT Glue Response" and "External Information" to differentiate the sources.
                5. **Insufficient Context**: If the provided context does not contain enough information to answer the question, state: "The provided context does not contain enough information to answer the question."
                6. **Document References**: List the names of all documents accessed, along with a confidence score for each document based on its relevance. 

                ### Example ###

                #### IT Glue Response ####
                [Your answer based on the given context]
                                                       
                ## External Information ##
                []

                #### Alerts and Escalation Matrix ####
                [Your answer after referring to the specific escalation matrix file for the given account name to determine who should be alerted or to whom the issue should be escalated. Ensure that the appropriate contacts are notified based on the escalation matrix provided in the document.] 
                
                ### Document Names ###
                [List of documents and confidence scores (in %) with descending order.] 
                
                **Answer:**
                """))
                run_chatbot(faiss_indexes, blob_service_client, embeddings, qa_chain)
            
            elif selected == 'Alerts and Escalation Matrix':
                with st.sidebar:
                    client_names = ["Select an Account Name"] + account_names
                    selected_client = st.selectbox("**Select Account Name** 🚩", client_names)
                    if selected_client != st.session_state['previous_clientOrg']:
                        st.session_state['clientOrg'] = selected_client
                        st.session_state['previous_clientOrg'] = selected_client

                    if st.session_state['clientOrg'] and st.session_state['clientOrg'] != "Select an Account Name":
                        st.info(f"You are viewing the {st.session_state['clientOrg']} Account")

                if st.session_state['clientOrg'] and st.session_state['clientOrg'] != "Select an Account Name":
                    alerts_content = load_content_from_blob(blob_service_client, container_name, st.session_state['clientOrg'], "Alerts")
                    escalation_matrix_content = load_content_from_blob(blob_service_client, container_name, st.session_state['clientOrg'], "Escalation Matrix")

                    st.subheader("Alerts")
                    st.markdown(
                        f"""
                        <div style="
                            border: 2px solid #ffcc00; 
                            padding: 15px; 
                            border-radius: 10px; 
                            background-color: #fff7e6;">
                            {alerts_content}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.subheader("Escalation Matrix")
                    st.markdown(
                        f"""
                        <div style="
                            border: 2px solid #0066cc; 
                            padding: 15px; 
                            border-radius: 10px; 
                            background-color: #e6f2ff;">
                            {escalation_matrix_content}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
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
