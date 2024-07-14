import base64
from io import BytesIO
import os
from openai import OpenAI
import pandas as pd
from config import OPENAI_API_KEY
from .logging_service import write_log_to_file, write_results_to_csv
import time

openai = OpenAI(api_key=OPENAI_API_KEY)

def classify_with_gpt_3_5_turbo_zero_shot(text, categories, temperature, classification_method, true_category):
    return classify_inquiry_zero_shot(text, categories, model="gpt-3.5-turbo-0125", classification_type="zero_shot", temperature=temperature, classification_method=classification_method, true_category=true_category)

def classify_with_gpt_3_5_turbo_few_shot(text, categories, temperature, classification_method, true_category):
    return classify_inquiry_few_shot(text, categories, model="gpt-3.5-turbo-0125", classification_type="few_shot", temperature=temperature, classification_method=classification_method, true_category=true_category)

def classify_with_gpt_3_5_turbo_fine_zero_shot(text, categories, temperature, classification_method, true_category):
    return classify_inquiry_zero_shot(text, categories, model="ft:gpt-3.5-turbo-0125:personal:classifier-model:9k0H1hdV", classification_type="fine_zero_shot", temperature=temperature, classification_method=classification_method, true_category=true_category)

def classify_with_gpt_3_5_turbo_fine_few_shot(text, categories, temperature, classification_method, true_category):
    return classify_inquiry_few_shot(text, categories, model="ft:gpt-3.5-turbo-0125:personal:classifier-model:9k0H1hdV", classification_type="fine_few_shot", temperature=temperature, classification_method=classification_method, true_category=true_category)

def classify_with_gpt_3_5_turbo_fine_no_prompting(text, temperature, classification_method, true_category):
    return classify_inquiry_no_prompting(text, model="ft:gpt-3.5-turbo-0125:personal:classifier-model:9k0H1hdV", classification_type="fine_no_prompting", temperature=temperature, classification_method=classification_method, true_category=true_category)

def classify_inquiry_zero_shot(text, categories, model, classification_type, temperature, classification_method, true_category):
    categories_str = ", ".join(categories)
    system_message = f"You are a Customer Service Inquiry classifier. Classify the following Inquiry into one of these categories: {categories_str}. Respond only with the category name."
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Text: '{text}'"}
    ]

    response = retry_operation(openai.chat.completions.create, model=model, messages=messages, max_tokens=20, temperature=temperature, n=1, stop=None)

    classification = response.choices[0].message.content.strip()
    usage = response.usage

    write_log_to_file(model, classification_type, classification_method, temperature, system_message, text, true_category, classification, usage)
    write_results_to_csv(model, classification_type, classification_method, temperature, text, true_category, classification, usage)
    return classification

def classify_inquiry_few_shot(text, categories, model, classification_type, temperature, classification_method, true_category):
    categories_str = ", ".join(categories)
    examples = [
        {"text": "How do I locate my card?", "category": "card_arrival"},
        {"text": "I'm trying to figure out the current exchange rate.", "category": "exchange_rate"},
        {"text": "Why am I being charged more?", "category": "card_payment_wrong_exchange_rate"},
        {"text": "Send my card as soon as you are able to.", "category": "card_delivery_estimate"},
        {"text": "What can you do to unblock my pin?", "category": "pin_blocked"}
    ]
    
    examples_str = "\n".join([f"Example {i+1} - Inquiry: '{example['text']}' Answer: '{example['category']}'" for i, example in enumerate(examples)])
    system_message = f"You are a Customer Service Inquiry classifier. Classify the following Inquiry into one of these categories: {categories_str}. Here are some examples:\n{examples_str}. Respond only with the category name."
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Text: '{text}'"}
    ]
    
    response = retry_operation(openai.chat.completions.create, model=model, messages=messages, max_tokens=20, temperature=temperature, n=1, stop=None)

    classification = response.choices[0].message.content.strip()
    usage = response.usage

    write_log_to_file(model, classification_type, classification_method, temperature, system_message, text, true_category, classification, usage)
    write_results_to_csv(model, classification_type, classification_method, temperature, text, true_category, classification, usage)
    return classification

def classify_inquiry_no_prompting(text, model, classification_type, temperature, classification_method, true_category):
    system_message = f"You are a Customer Service Inquiry classifier. Classify the following Inquiry. Respond only with the category name."
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Text: '{text}'"}
    ]

    response = retry_operation(openai.chat.completions.create, model=model, messages=messages, max_tokens=20, temperature=temperature, n=1, stop=None)

    classification = response.choices[0].message.content.strip()
    usage = response.usage

    write_log_to_file(model, classification_type, classification_method, temperature, system_message, text, true_category, classification, usage)
    write_results_to_csv(model, classification_type, classification_method, temperature, text, true_category, classification, usage)
    return classification

def upload_file_to_openai(file_path):
    with open(file_path, 'rb') as file:
        response = openai.files.create(
            file=file,
            purpose='fine-tune'
        )
    return response

def fine_tune_model(training_file_id, model='gpt-3.5-turbo-0125', n_epochs=4):
    response = openai.fine_tuning.jobs.create(
        training_file=training_file_id,
        model=model,
        suffix="classifier-model",
        hyperparameters={
            "n_epochs":2,
            "learning_rate_multiplier":2.0,
            "batch_size": 1
        }
    )
    return response

def get_fine_tune_status(fine_tuning_job_id):
    response = openai.fine_tuning.jobs.retrieve(fine_tuning_job_id=fine_tuning_job_id)
    return response

def list_fine_tune_jobs():
    response = openai.fine_tuning.jobs.list()
    return response.data

def retry_operation(function, *args, **kwargs):
    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            return function(*args, **kwargs)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                print("Retrying in {} seconds...".format(retry_delay))
                time.sleep(retry_delay)
            else:
                print("All retry attempts failed.")
                raise

# Function to retrieve fine-tuning job metrics
def get_fine_tuning_metrics(fine_tune_id, base_results_dir='./customer_service_classifier_ai_data/results', folder_name='training'):
    try:
        # Create the directory if it does not exist
        output_dir = os.path.join(base_results_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)

        # Set the output CSV file path
        output_csv = os.path.join(output_dir, 'fine_tuning_metrics.csv')

        # Check if the file already exists
        if os.path.exists(output_csv):
            return

        # Retrieve fine-tuning job details
        fine_tune = openai.fine_tuning.jobs.retrieve(fine_tune_id)

        # Print fine-tuning job details
        print("Fine-tuning job details:")
        print(f"ID: {fine_tune.id}")
        print(f"Status: {fine_tune.status}")
        print(f"Model: {fine_tune.model}")
        print(f"Created At: {fine_tune.created_at}")
        print(f"Updated At: {fine_tune.finished_at}")
        print(f"Hyperparameters: {fine_tune.hyperparameters}")

        # If the fine-tuning job is completed, retrieve the result files
        if fine_tune.status == 'succeeded':
            for result_file in fine_tune.result_files:
                result_file_content = openai.files.content(file_id=result_file)
                content_bytes = result_file_content.read()

                # Decode the base64 content
                decoded_content = base64.b64decode(content_bytes)

                # Convert the decoded content to a pandas DataFrame
                result_df = pd.read_csv(BytesIO(decoded_content))
                result_df.to_csv(output_csv, index=False)
                
    except Exception as e:
        print(f"Error retrieving fine-tuning job metrics: {e}")