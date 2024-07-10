from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Create the CustomerServiceClassifierAI data folder
folder_path = './customer_service_classifier_ai_data'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)