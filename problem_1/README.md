# Airline Call Center Optimization System

## Installation


## Create a virtual environment:
   python -m venv myenv


## Activate the virtual environment:
   - Windows:
      myenv\Scripts\activate
    
   - Mac/Linux:
      source myenv/bin/activate
    

## Install dependencies:
   pip install -r requirements.txt

## Set up API keys:
   - Create a `.env` file in the root directory.
   - Add your Together AI API key:
     TOGETHER_API_KEY=your_api_key_here

## Running the Code

1. Run the main script:
   python main.py

2. Interact with the system:
   - The script will process simulated calls and provide AI-generated responses.
   - You can also manually enter flight inquiries when prompted.

## Multi-Agent Function Calling Approach

The system consists of two AI agents working together:

### 1. Info Agent
- Fetches structured flight data from a predefined dataset.
- Responds strictly in JSON format.

### 2. QA Agent
- Extracts flight numbers from user queries.
- Calls the Info Agent to retrieve flight information.
- Formats the response in a structured JSON output for the user.

## Features
- Flight inquiry handling with structured responses.
- Audio transcription simulation for customer interactions.
- Categorization of customer issues (e.g., refund, cancellation, complaint).
- AI-powered customer service responses using Together AI.
- KPI computation to analyze call center performance.

## Notes
- Ensure your API key is correctly set up in the `.env` file before running.
- Modify `airline_flights` in `main.py` to include more flights if needed.


