import urllib.request
import json

def post(url, data):
    req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())

def get(url):
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode())

print("=== 1. TESTING GET /api/samples ===")
samples = get('http://127.0.0.1:5000/api/samples')
print(f"Retrieved {len(samples)} sample complaints.")

print("\n=== 2. TESTING GET /api/metrics ===")
metrics = get('http://127.0.0.1:5000/api/metrics')
print(f"Dataset size: {metrics['dataset_size']}")
print(f"Category Model Accuracy: {metrics['category_accuracy']}%")
print(f"Priority Model Accuracy: {metrics['priority_accuracy']}%")

print("\n=== 3. TESTING 10 COMPLAINT ANALYSES ===")
test_cases = [
    "Water is leaking continuously from the ceiling of Block B classroom 204.",
    "Sparks flying from the main switchboard near Room 301, major electrical hazard!",
    "The classroom projector is not working in Room 302.",
    "Wi-Fi network signal dropping constantly in Hostel 3 wing B since morning.",
    "Hostel room 304 door lock jammed, student trapped inside room!",
    "Chemical spill of concentrated acid in chemistry lab 2 floor!",
    "Unidentified intruder spotted near girl hostel back perimeter wall!",
    "Washrooms in Block B ground floor extremely dirty and unhygienic.",
    "Bus 4 breakdown near east gate blocking main exit road.",
    "Inquiry regarding lost student wallet containing ID cards found near main quad."
]

for idx, text in enumerate(test_cases, 1):
    res = post('http://127.0.0.1:5000/api/analyze', {'complaint': text})
    print(f"[{idx}] Input: '{text}'")
    print(f"     Cat: {res['category']} | Prio: {res['priority']} | Dept: {res['department']}")
    print(f"     Loc: {res['location']} | Obj: {res['object']} | Conf: {res['confidence']}%\n")
