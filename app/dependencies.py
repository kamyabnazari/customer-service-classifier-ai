from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from app.models.database import create_tables
from databases import Database
from sqlalchemy import create_engine
import logging
import os

# Create the CustomerServiceClassifierAI data folder
folder_path = './customer_service_classifier_ai_data'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Create database instance
DATABASE_URL = "sqlite:///./customer_service_classifier_ai_data/customer_service_classifier_ai.db"
database = Database(DATABASE_URL)
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

@asynccontextmanager
async def lifespan(app):
    try:
        await database.connect()
        create_tables(engine)
    except Exception as e:
        logging.error(f"Failed to connect to the database: {e}")
        raise
    yield
    try:
        await database.disconnect()
    except Exception as e:
        logging.error(f"Failed to disconnect from the database: {e}")
        raise

def configure_cors(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

def get_database():
    return database