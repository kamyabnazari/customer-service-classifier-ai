# CustomerServiceClassifierAI

<div style="text-align: center;">
  <img src="misc/customer_service_classifier_ai_icon.png" alt="CustomerServiceClassifierAI Framework Logo" style="max-width: 30%; height: auto; display: block; margin: 0 auto;">
</div>

## Introduction

CustomerServiceClassifierAI is an advanced research project exploring the effectiveness of Large Language Models (LLMs) in classifying customer inquiries in the banking sector. This project employs experimental approaches with prompt-engineering and model fine-tuning techniques, aiming to enhance customer service through precise inquiry classification.

<div style="text-align: center;">
  <img src="misc/application_screenshot.png" alt="Application Screenshot" style="max-width: 30%; height: auto; display: block; margin: 0 auto;">
</div>

## Features

- Utilizes (BANKING77) a high-quality labeled dataset of banking customer inquiries to verify classification accuracy.
- Implements Zero-Shot and Few-Shot prompting to optimize response quality with no or minimal example inputs.
- Adjusts the LLM using specific dataset segments to increase the precision of model predictions.
- Uses standardized machine learning metrics to objectively assess prototype performance.
- Assesses the cost-effectiveness of the employed techniques.
- Used Models are OpenAIs GPT-3.5 Turbo, GPT-4 Turbo, GPT-4o

## Objective:

The project aims to enhance customer interactions and provide strategic insights into the feasibility and effectiveness of LLMs in customer communication using cutting-edge AI technologies. It targets developers, researchers, and customer service professionals looking to innovate at the technology frontier.

## Getting Started

### Tech-Stack

#### General

- [Git](https://git-scm.com) [Version Control]
- [GitHub](https://github.com/) [Code Hosting]

#### Application

- [Python](https://www.python.org/downloads/) [Programming Language]
- [Streamlit](https://www.typescriptlang.org/) [Programming Language]
- [OpenAI GPT](https://openai.com/) [Language Processing]

### Prerequisites

- [Git](https://git-scm.com/downloads)
- [Python](https://www.python.org/downloads/)

---

### Installation

#### Clone

```bash
https://github.com/kamyabnazari/customer-service-classifier-ai
```

#### Setup

### Environment variables

Create a `.env` file in the root of the backend directory with the following variables:

```
OPENAI_API_KEY={your openai api key}
```

### Running the backend

To install the required packages for this service, run the following commands:

```
pip install -r requirements.txt

or

pip3 install -r requirements.txt
```

To run this service, run the following commands:

```
streamlit run app/main.py
```

---

### License

This project is licensed under the [Attribution 4.0 International License] License - see the LICENSE file for details.

---

### Acknowledgments

Special thanks to the OpenAI team for providing the API that powers our intelligent agents.

Additionally for the dataset by [PolyAI]("https://github.com/PolyAI-LDN/task-specific-datasets")
