from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

openai = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
    )

# Create the CustomerServiceClassifierAI data folder
folder_path = './customer_service_classifier_ai_data'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)