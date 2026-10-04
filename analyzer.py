import os
import re
import json
import joblib
import numpy as np

class ComplaintAnalyzer:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.cat_model = None
        self.prio_model = None
        self.vectorizer = None
        self.metrics = {}
        self.load_models()

    def load_models(self):
        cat_path = os.path.join(self.models_dir, "category_model.pkl")
        prio_path = os.path.join(self.models_dir, "priority_model.pkl")
        vec_path = os.path.join(self.models_dir, "tfidf_vectorizer.pkl")
        metrics_path = os.path.join(self.models_dir, "metrics.json")

        if os.path.exists(cat_path) and os.path.exists(prio_path) and os.path.exists(vec_path):
            self.cat_model = joblib.load(cat_path)
            self.prio_model = joblib.load(prio_path)
            self.vectorizer = joblib.load(vec_path)
        else:
            raise FileNotFoundError("Model files not found. Please run train_model.py first.")

        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                self.metrics = json.load(f)

    def preprocess(self, text):
        t = str(text).lower()
        t = re.sub(r'[^a-z0-9\s]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    def extract_information(self, text):
        """Rule-based and pattern matching lightweight NLP entity extraction."""
        t_clean = text.strip()

        # 1. Location Extraction
        location_patterns = [
            r'\b(?:block|building|hostel)\s+[a-z0-9\-]+(?:\s+(?:room|rm|classroom|lab)\s+[a-z0-9\-]+)?\b',
            r'\b(?:room|rm|classroom|lab|seminar hall|hall|auditorium|office)\s+[a-z0-9]*\d+[a-z0-9]*\b',
            r'\b(?:1st|2nd|3rd|4th|5th|ground|top|basement)\s+floor\b',
            r'\b(?:cafeteria|canteen|library|sports ground|main gate|east gate|north gate|parking lot|admin block|reception|mess)\b'
        ]

        locations_found = []
        for pat in location_patterns:
            matches = re.findall(pat, t_clean, flags=re.IGNORECASE)
            for m in matches:
                m_clean = m.strip()
                if m_clean.lower() not in [loc.lower() for loc in locations_found]:
                    locations_found.append(m_clean.title())

        # Deduplicate substrings (e.g. if "Block B Classroom 204" contains "Classroom 204")
        final_locations = []
        for loc in locations_found:
            if not any(loc != other and loc in other for other in locations_found):
                final_locations.append(loc)

        location_str = ", ".join(final_locations) if final_locations else "Unspecified Location"

        # 2. Equipment / Object Extraction
        equipment_keywords = [
            "projector", "fan", "ceiling fan", "tubelight", "light", "wire", "switch", "switchboard",
            "air conditioner", "ac", "geyser", "pipe", "tap", "sink", "washbasin", "drain", "water tank",
            "flush", "door", "door lock", "window", "staircase", "elevator", "lift", "desk", "chair", "podium",
            "smartboard", "whiteboard", "router", "wi-fi", "wifi", "pc", "computer", "desktop", "monitor",
            "keyboard", "mouse", "printer", "mic", "microphone", "amplifier", "shuttle", "bus", "ev charger",
            "cctv", "camera", "lockbook", "biometric", "trash bin", "dustbin", "soap dispenser", "water cooler"
        ]

        objects_found = []
        for eq in equipment_keywords:
            if re.search(r'\b' + re.escape(eq) + r'\b', t_clean, re.IGNORECASE):
                objects_found.append(eq.title())

        object_str = ", ".join(objects_found[:3]) if objects_found else "General Facility Item"

        # 3. Time Reference Extraction
        time_patterns = [
            r'\b(?:since|for|past|last)\s+(?:\d+|a|an|the)?\s*(?:minutes?|hours?|days?|weeks?|months?|yesterday|morning|night)\b',
            r'\b(?:yesterday|this morning|today|tonight|overnight|recently)\b'
        ]
        time_refs = []
        for tp in time_patterns:
            matches = re.findall(tp, t_clean, flags=re.IGNORECASE)
            time_refs.extend(matches)
        time_str = ", ".join(time_refs) if time_refs else "Not specified"

        # 4. Identified Problem Phrase
        problem_triggers = [
            "sparks flying", "short circuit", "burnt", "fire", "leakage", "leaking", "burst", "flooding",
            "broken", "not working", "stopped working", "offline", "unresponsive", "damaged", "intruder",
            "overflowing", "dirty", "infestation", "clash", "missing", "brake failure", "overcrowded"
        ]
        prob_found = []
        for pt in problem_triggers:
            if pt in t_clean.lower():
                prob_found.append(pt.title())
        problem_str = ", ".join(prob_found[:2]) if prob_found else "Issue reported"

        # 5. Extract Keywords / Reasoning Signals
        cleaned = self.preprocess(t_clean)
        vec_features = self.vectorizer.transform([cleaned])
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        row_vec = vec_features.toarray()[0]
        nonzero_idx = np.where(row_vec > 0)[0]
        sorted_idx = nonzero_idx[np.argsort(row_vec[nonzero_idx])[::-1]]

        reasoning_signals = list(feature_names[sorted_idx[:6]])
        if not reasoning_signals:
            reasoning_signals = [w for w in cleaned.split() if len(w) > 3][:5]

        return {
            "location": location_str,
            "object": object_str,
            "problem": problem_str,
            "time_reference": time_str,
            "reasoning_signals": reasoning_signals
        }

    def predict(self, text):
        if not text or len(text.strip()) == 0:
            return {
                "error": "Empty complaint text provided."
            }

        cleaned = self.preprocess(text)
        vec = self.vectorizer.transform([cleaned])

        # Category Prediction
        cat_probs = self.cat_model.predict_proba(vec)[0]
        cat_idx = np.argmax(cat_probs)
        category = self.cat_model.classes_[cat_idx]
        confidence = float(cat_probs[cat_idx])

        # Priority Prediction (ML Model)
        prio_probs = self.prio_model.predict_proba(vec)[0]
        prio_idx = np.argmax(prio_probs)
        raw_priority = self.prio_model.classes_[prio_idx]

        # Rule-based safety overrides for high-hazard keywords
        text_lower = text.lower()

        critical_triggers = [
            "fire", "sparks flying", "gas leak", "intruder", "short circuit", "burst", "live wire",
            "brake failure", "trapped", "slab collapsed", "ransomware", "severe sewage flood",
            "chemical spill", "acid spill", "biohazard", "explosion"
        ]
        high_triggers = [
            "leaking continuously", "flooding", "blackout", "no water supply", "cctv offline",
            "stolen", "infestation", "reckless driving", "dangerous", "power outage",
            "extremely dirty", "unhygienic", "breakdown", "blocking"
        ]

        final_priority = raw_priority
        if any(trig in text_lower for trig in critical_triggers):
            final_priority = "CRITICAL"
        elif any(trig in text_lower for trig in high_triggers) and raw_priority not in ["CRITICAL", "HIGH"]:
            final_priority = "HIGH"

        # Map Severity
        severity_map = {
            "CRITICAL": "CRITICAL",
            "HIGH": "HIGH",
            "MEDIUM": "MEDIUM",
            "LOW": "LOW"
        }
        severity = severity_map.get(final_priority, "MEDIUM")

        # Department Mapping
        dept_map = {
            "Electrical": "Electrical Works",
            "Plumbing": "Maintenance & Plumbing",
            "Infrastructure": "Estate & Facilities",
            "IT / Technical": "IT Support Desk",
            "Transportation": "Fleet & Transit",
            "Security": "Campus Security",
            "Cleaning": "Sanitation & Hygiene",
            "Academic": "Academic Affairs",
            "Hostel": "Hostel Management",
            "Other": "Administrative Services"
        }
        department = dept_map.get(category, "Administrative Services")

        # Extract NLP entities
        extracted_info = self.extract_information(text)

        # Recommended Action
        action_map = {
            "CRITICAL": f"IMMEDIATE EMERGENCY DISPATCH: Notify {department} shift engineer and campus emergency response immediately.",
            "HIGH": f"PRIORITY WORK ORDER: Assign senior technician from {department} for inspection within 2 hours.",
            "MEDIUM": f"STANDARD WORK ORDER: Ticket logged with {department}. Resolution targeted within 24 hours.",
            "LOW": f"ROUTINE MAINTENANCE: Logged in {department} queue for scheduled preventive maintenance."
        }

        return {
            "complaint_text": text,
            "category": category,
            "priority": final_priority,
            "severity": severity,
            "department": department,
            "confidence": round(confidence * 100, 1),
            "raw_ml_priority": raw_priority,
            "location": extracted_info["location"],
            "object": extracted_info["object"],
            "problem": extracted_info["problem"],
            "time_reference": extracted_info["time_reference"],
            "reasoning_signals": extracted_info["reasoning_signals"],
            "recommended_action": action_map.get(final_priority, "Process ticket.")
        }
