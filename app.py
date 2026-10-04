import os
import json
from flask import Flask, render_template, request, jsonify
from analyzer import ComplaintAnalyzer

app = Flask(__name__)

# Initialize Complaint Analyzer
analyzer = None
try:
    analyzer = ComplaintAnalyzer()
    print("Complaint Analyzer initialized successfully.")
except Exception as e:
    print(f"Error initializing analyzer: {e}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def analyze_complaint():
    if not analyzer:
        return jsonify({"error": "Model not loaded. Please run train_model.py first."}), 500

    data = request.get_json() or {}
    complaint_text = data.get("complaint", "").strip()

    if not complaint_text:
        return jsonify({"error": "Please enter a complaint description."}), 400

    result = analyzer.predict(complaint_text)
    return jsonify(result)

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    metrics_path = os.path.join("models", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({"error": "Metrics file not found."}), 404

@app.route("/api/samples", methods=["GET"])
def get_samples():
    samples = [
        {
            "title": "Water Leakage",
            "text": "Water is leaking continuously from the ceiling of Block B classroom 204.",
            "expected_prio": "HIGH"
        },
        {
            "title": "Fire & Electrical Hazard",
            "text": "Sparks flying from the main switchboard near Room 301, major electrical hazard!",
            "expected_prio": "CRITICAL"
        },
        {
            "title": "Projector Not Working",
            "text": "The classroom projector is not working in Room 302.",
            "expected_prio": "MEDIUM"
        },
        {
            "title": "Wi-Fi Outage",
            "text": "Wi-Fi network signal dropping constantly in Hostel 3 wing B since morning.",
            "expected_prio": "HIGH"
        },
        {
            "title": "Trapped in Hostel Room",
            "text": "Hostel room 304 door lock jammed, student trapped inside room!",
            "expected_prio": "CRITICAL"
        },
        {
            "title": "Dirty Restrooms",
            "text": "Washrooms in Block B ground floor extremely dirty and unhygienic.",
            "expected_prio": "HIGH"
        },
        {
            "title": "Flickering Light",
            "text": "Tubelight flickering continuously in Room 108 Block C.",
            "expected_prio": "MEDIUM"
        },
        {
            "title": "Lost Wallet Query",
            "text": "Inquiry regarding lost student wallet containing ID cards found near main quad.",
            "expected_prio": "LOW"
        }
    ]
    return jsonify(samples)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
