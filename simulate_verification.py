import requests
import time
import json

BASE_URL = "http://localhost:5000"
API_KEY = "lab_2026_secure_key"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

def test_health():
    print("Testing health...")
    r = requests.get(f"{BASE_URL}/api/health")
    print(r.json())

def send_event(motion, door, hour=None):
    print(f"Sending event: motion={motion}, door={door}, hour={hour}...")
    payload = {
        "device_id": "test_script",
        "motion": motion,
        "door": door,
        "time": f"2026-02-19T{hour}:00:00" if hour is not None else None
    }
    r = requests.post(f"{BASE_URL}/event", json=payload, headers=HEADERS)
    print(r.json())
    return r.json()

def test_off_hours():
    # Test 2:00 AM (Should be intrusion if armed)
    print("\n--- Testing Off-Hours (02:00) ---")
    send_event(motion=1, door=0, hour="02")
    
    # Test 12:00 PM (Should be normal even if armed)
    print("\n--- Testing Working Hours (12:00) ---")
    send_event(motion=1, door=0, hour="12")

def test_label_and_retrain():
    print("\n--- Testing Label and Retrain ---")
    # Get last event
    r = requests.get(f"{BASE_URL}/api/events?limit=1")
    events = r.json()
    if not events:
        print("No events found to label.")
        return
    
    event_id = events[0]['id']
    print(f"Labeling event {event_id} as normal...")
    r = requests.post(f"{BASE_URL}/api/label/{event_id}", json={"label": "normal"}, headers=HEADERS)
    print(r.json())

if __name__ == "__main__":
    try:
        test_health()
        test_off_hours()
        test_label_and_retrain()
    except Exception as e:
        print(f"Error: {e}. Is the server running?")
