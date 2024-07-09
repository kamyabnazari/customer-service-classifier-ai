from models.database import create_tables
from databases import Database
from sqlalchemy import create_engine
import os

# Create the CustomerServiceClassifierAI data folder
folder_path = './customer_service_classifier_ai_data'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Create database instance
DATABASE_URL = "sqlite:///./customer_service_classifier_ai_data/customer_service_classifier_ai.db"
database = Database(DATABASE_URL)
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

create_tables(engine)

def get_database():
    return database