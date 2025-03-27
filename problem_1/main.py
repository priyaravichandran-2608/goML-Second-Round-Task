import os
import json
import re
import random
from typing import List, Dict, Optional
import together
from dotenv import load_dotenv

load_dotenv('api_keys.env')  # Explicitly load from api_keys.env

# Retrieve Together API key
api_key = os.getenv("TOGETHER_API_KEY")
if not api_key:
    raise ValueError(" Critical Error: API key not found in api_keys.env")
os.environ["TOGETHER_API_KEY"] = api_key

# Initialize Together client
together.api_key = api_key

def get_flight_info(flight_number: str) -> Optional[Dict[str, str]]:
 
    flights = {
        "AI123": {
            "flight_number": "AI123",
            "departure_time": "08:00 AM",
            "destination": "Delhi",
            "status": "Delayed"
        },
        "AI456": {
            "flight_number": "AI456",
            "departure_time": "10:00 AM",
            "destination": "Mumbai",
            "status": "On Time"
        },
        "AI789": {
            "flight_number": "AI789",
            "departure_time": "12:00 PM",
            "destination": "Bangalore",
            "status": "On Time"
        }
    }
    return flights.get(flight_number.upper(), None)

def info_agent_request(flight_number: str) -> str:
  
    flight_info = get_flight_info(flight_number)
    if flight_info is None:
        return json.dumps({"error": f"Flight {flight_number} not found in database."})
    return json.dumps(flight_info)

def qa_agent_respond(user_query: str) -> str:

    # Extract flight number using a more robust regex
    match = re.search(r'(?:Flight|flight|FLIGHT)\s*([A-Za-z0-9]+)', user_query, re.IGNORECASE)
    if not match:
        return json.dumps({"answer": "No valid flight number found in query."})
    
    flight_number = match.group(1)
    flight_info_json = info_agent_request(flight_number)
    flight_info = json.loads(flight_info_json)
    
    if "error" in flight_info:
        return json.dumps({"answer": flight_info["error"]})
    
    answer = (
        f"Flight {flight_info['flight_number']} departs at {flight_info['departure_time']} "
        f"to {flight_info['destination']}. Current status: {flight_info['status']}."
    )
    return json.dumps({"answer": answer})

# Audio Transcription Functionality


def transcribe_audio(audio_file: str) -> str:
   
    simulated_transcripts = {
        "call1.wav": "I would like to cancel my flight due to a personal emergency.",
        "call2.wav": "I want to get a refund for my flight ticket. The flight AI123 was delayed.",
        "call3.wav": "The service was excellent but my seat was uncomfortable.",
        "call4.wav": "Can you tell me what time Flight AI456 departs?",
        "call5.wav": "What is the status of Flight AI789?"
    }
    return simulated_transcripts.get(audio_file, "Audio transcription not available.")

# Issue Categorization Functionality


def categorize_issue(transcript: str) -> str:
    
    transcript_lower = transcript.lower()
    if "cancel" in transcript_lower:
        return "Flight Cancellation"
    elif "refund" in transcript_lower:
        return "Refund"
    elif re.search(r'(?:flight|Flight|FLIGHT)\s*[A-Za-z0-9]+', transcript_lower):
        if "what time" in transcript_lower or "depart" in transcript_lower or "status" in transcript_lower:
            return "Flight Inquiry"
    elif "complaint" in transcript_lower or "unhappy" in transcript_lower or "uncomfortable" in transcript_lower:
        return "Complaint"
    else:
        return "General Inquiry"

# Together API Generative Functionality


def together_analyze_call(transcript: str) -> str:
  
    try:
        prompt = (
            f"You are a customer service agent at an airport service center. A customer has called with the following concern: '{transcript}'. "
            "Please provide a helpful and professional response addressing their concern directly. "
            "If the issue is about flight information, provide accurate details. "
            "If it's a complaint or request, offer a solution or next steps."
        )
        response = together.Completion.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return json.dumps({"error": f"Together API error: {str(e)}"})

# KPI Computation Functionality


def compute_kpis(call_data: List[Dict]) -> Dict:
  
    if not call_data:
        return {"error": "No call data provided."}
    
    total_calls = len(call_data)
    total_duration = sum(call.get("duration", 0) for call in call_data)
    average_duration = total_duration / total_calls if total_calls else 0
    resolved_calls = sum(1 for call in call_data if call.get("resolved", False))
    resolution_rate = (resolved_calls / total_calls) * 100 if total_calls else 0
    
    return {
        "total_calls": total_calls,
        "average_call_duration_sec": round(average_duration, 2),
        "resolution_rate_percent": round(resolution_rate, 2)
    }

# Call Processing Workflow


def process_call(audio_file: str) -> None:
   
    print("=== Incoming Call ===")
    print(f"Receiving audio: {audio_file}")
    
    # Transcription
    transcript = transcribe_audio(audio_file)
    print(f"Transcript: {transcript}")
    
    # Categorization
    category = categorize_issue(transcript)
    print(f"Call Category: {category}")
    
    # Process based on category
    if category == "Flight Inquiry":
        response = qa_agent_respond(transcript)
        print("Flight Inquiry Response:", response)
    elif category in ["Flight Cancellation", "Refund"]:
        print(f"Triggering {category} process. (Further integration needed.)")
    else:
        analysis = together_analyze_call(transcript)
        print("Customer Service Response:", analysis)
    
    print("---------------------------\n")

# Main Simulation


if __name__ == "__main__":
    # Process simulated audio calls
    simulated_audio_files = ["call1.wav", "call2.wav", "call3.wav", "call4.wav", "call5.wav"]
    for audio in simulated_audio_files:
        process_call(audio)
    
    # Run tests for the two-agent flight inquiry system
    print("=== Two-Agent Flight Inquiry Tests ===")
    test_cases = [
        ("get_flight_info('AI123')", get_flight_info("AI123")),
        ("info_agent_request('AI123')", info_agent_request("AI123")),
        ("qa_agent_respond('When does Flight AI123 depart?')", qa_agent_respond("When does Flight AI123 depart?")),
        ("qa_agent_respond('What is the status of Flight AI999?')", qa_agent_respond("What is the status of Flight AI999?"))
    ]
    
    for test_name, test_result in test_cases:
        print(f"\nTest {test_name}:")
        print(test_result)
    
    # Simulate KPI computation over a series of calls
    print("\n=== KPI Computation ===")
    simulated_call_data = [
        {"duration": random.randint(200, 400), "resolved": True},
        {"duration": random.randint(200, 400), "resolved": False},
        {"duration": random.randint(200, 400), "resolved": True},
        {"duration": random.randint(200, 400), "resolved": True},
        {"duration": random.randint(200, 400), "resolved": False}
    ]
    kpis = compute_kpis(simulated_call_data)
    print("Simulated Call Data:", simulated_call_data)
    print("KPIs:", json.dumps(kpis, indent=2))