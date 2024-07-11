from openai import OpenAI
from config import OPENAI_API_KEY
from .logging_service import write_log_to_file, write_results_to_csv

# Set up the OpenAI API client
openai = OpenAI(api_key=OPENAI_API_KEY)

def classify_with_gpt_3_5_turbo_zero_shot(text, categories, temperature, classification_method):
    return classify_inquiry_zero_shot(text, categories, model="gpt-3.5-turbo-0125", classification_type="zero_shot", temperature=temperature, classification_method=classification_method)

def classify_with_gpt_3_5_turbo_few_shot(text, categories, temperature, classification_method):
    return classify_inquiry_few_shot(text, categories, model="gpt-3.5-turbo-0125", classification_type="few_shot", temperature=temperature,classification_method=classification_method)

def classify_with_gpt_3_5_turbo_fine_zero_shot(text, categories, temperature, classification_method):
    return classify_inquiry_zero_shot(text, categories, model="ft:gpt-3.5-turbo-0125:personal:classifier-model:9jq8ioDW", classification_type="fine_zero_shot", temperature=temperature,classification_method=classification_method)

def classify_with_gpt_3_5_turbo_fine_few_shot(text, categories, temperature, classification_method):
    return classify_inquiry_few_shot(text, categories, model="ft:gpt-3.5-turbo-0125:personal:classifier-model:9jq8ioDW", classification_type="fine_few_shot", temperature=temperature,classification_method=classification_method)

def classify_with_gpt_3_5_turbo_fine_no_prompting(text, temperature, classification_method):
    return classify_inquiry_no_prompting(text, model="ft:gpt-3.5-turbo-0125:personal:classifier-model:9jq8ioDW", classification_type="fine_few_shot", temperature=temperature,classification_method=classification_method)

def classify_inquiry_zero_shot(text, categories, model, classification_type, temperature, classification_method):
    categories_str = ", ".join(categories)
    system_message = f"You are a Customer Service Inquiry classifier. Classify the following Inquiry into one of these categories: {categories_str}. Respond only with the category name."
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Text: '{text}'"}
    ]

    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=20,
        temperature=temperature,
        n=1,
        stop=None
    )

    classification = response.choices[0].message.content.strip()
    usage = response.usage

    write_log_to_file(model, classification_type, classification_method, temperature, system_message, text, classification, usage)
    write_results_to_csv(model, classification_type, classification_method, temperature, text, classification, usage)
    return classification

def classify_inquiry_few_shot(text, categories, model, classification_type, temperature, classification_method):
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
    
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=20,
        temperature=temperature,
        n=1,
        stop=None
    )

    classification = response.choices[0].message.content.strip()
    usage = response.usage

    write_log_to_file(model, classification_type, classification_method, temperature, system_message, text, classification, usage)
    write_results_to_csv(model, classification_type, classification_method, temperature, text, classification, usage)
    return classification

def classify_inquiry_no_prompting(text, model, classification_type, temperature, classification_method):  
    system_message = f"Classify the following Inquiry respond only with the category name."
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Text: '{text}'"}
    ]
    
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=20,
        temperature=temperature,
        n=1,
        stop=None
    )

    classification = response.choices[0].message.content.strip()
    usage = response.usage

    write_log_to_file(model, classification_type, classification_method, temperature, system_message, text, classification, usage)
    write_results_to_csv(model, classification_type, classification_method, temperature, text, classification, usage)
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