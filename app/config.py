from dotenv import load_dotenv
import os

# Lade Umgebungsvariablen aus der .env-Datei
load_dotenv()

# Lese den OpenAI API-Schlüssel aus den Umgebungsvariablen
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Erstelle ein Verzeichnis für Anwendungsdaten, falls es nicht existiert
folder_path = './customer_service_classifier_ai_data'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
