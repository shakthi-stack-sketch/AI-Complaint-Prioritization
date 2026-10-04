import os
import json
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

DATASET_PATH = os.path.join("data", "complaints.csv")

def generate_dataset():
    """Generate a rich, multi-domain dataset of facility and academic complaints."""
    raw_data = [
        # --- ELECTRICAL ---
        ("Sparks flying from the main switchboard near Room 301, major electrical hazard!", "Electrical", "CRITICAL", "CRITICAL", "Electrical Works"),
        ("Exposed live wire hanging from ceiling in Block B corridor near room 104.", "Electrical", "CRITICAL", "CRITICAL", "Electrical Works"),
        ("Main circuit breaker panel smoking and smells burnt in Hostel 2 basement.", "Electrical", "CRITICAL", "CRITICAL", "Electrical Works"),
        ("Short circuit in lab 3 caused small fire near computer desk 12.", "Electrical", "CRITICAL", "CRITICAL", "Electrical Works"),
        ("Complete power outage in Science Block during final lab examination.", "Electrical", "HIGH", "HIGH", "Electrical Works"),
        ("Total blackout across Hostel Block 4 since 2 hours ago.", "Electrical", "HIGH", "HIGH", "Electrical Works"),
        ("Frequent voltage fluctuations causing server reboots in IT Lab 2.", "Electrical", "HIGH", "MEDIUM", "Electrical Works"),
        ("Ceiling fan making loud rattling noise and shaking violently in Room 204.", "Electrical", "MEDIUM", "MEDIUM", "Electrical Works"),
        ("Tubelight flickering continuously in Room 108 Block C.", "Electrical", "MEDIUM", "LOW", "Electrical Works"),
        ("Power socket on wall near desk 4 is loose and not delivering power.", "Electrical", "MEDIUM", "LOW", "Electrical Works"),
        ("Air conditioner in seminar hall is blowing warm air instead of cooling.", "Electrical", "MEDIUM", "MEDIUM", "Electrical Works"),
        ("Corridor lights on 3rd floor Block A are burnt out and need replacement.", "Electrical", "LOW", "LOW", "Electrical Works"),
        ("Wall switch for front light in Room 405 is broken.", "Electrical", "LOW", "LOW", "Electrical Works"),
        ("Exhaust fan in chemistry lab 1 is turning slowly.", "Electrical", "LOW", "LOW", "Electrical Works"),
        ("Indicator light on elevator panel is not glowing.", "Electrical", "LOW", "LOW", "Electrical Works"),
        ("Electrical socket near library reading table 6 is inactive.", "Electrical", "LOW", "LOW", "Electrical Works"),
        ("Ceiling fan speed controller knob is missing in Room 112.", "Electrical", "LOW", "LOW", "Electrical Works"),
        ("Light fixture glass shade cracked in admin lobby.", "Electrical", "LOW", "LOW", "Electrical Works"),

        # --- PLUMBING ---
        ("Gas leakage odor detected near chemistry lab pipe valves on floor 2!", "Plumbing", "CRITICAL", "CRITICAL", "Maintenance & Plumbing"),
        ("Severe sewage backup flooding ground floor hallway of Block A.", "Plumbing", "CRITICAL", "CRITICAL", "Maintenance & Plumbing"),
        ("Main water pipe burst flooding the basement storage room in library.", "Plumbing", "CRITICAL", "HIGH", "Maintenance & Plumbing"),
        ("Water leaking continuously from ceiling of Block B classroom 204.", "Plumbing", "HIGH", "HIGH", "Maintenance & Plumbing"),
        ("No water supply in Hostel Block 1 washrooms for past 6 hours.", "Plumbing", "HIGH", "HIGH", "Maintenance & Plumbing"),
        ("Overhead tank overflowing causing water stream on external staircase.", "Plumbing", "HIGH", "MEDIUM", "Maintenance & Plumbing"),
        ("Washbasin pipe broken and leaking heavy water onto washroom floor.", "Plumbing", "HIGH", "MEDIUM", "Maintenance & Plumbing"),
        ("Flush tank in 2nd floor mens restroom is continuously running and overflowing.", "Plumbing", "MEDIUM", "MEDIUM", "Maintenance & Plumbing"),
        ("Slow drainage in sink of hostel room 302 causing foul smell.", "Plumbing", "MEDIUM", "LOW", "Maintenance & Plumbing"),
        ("Tap leaking dripping water steadily in lab 4 pantry.", "Plumbing", "MEDIUM", "LOW", "Maintenance & Plumbing"),
        ("Hot water geyser not heating properly in Hostel 3 floor 2.", "Plumbing", "MEDIUM", "MEDIUM", "Maintenance & Plumbing"),
        ("Shower head broken in Hostel 2 bathroom 5.", "Plumbing", "LOW", "LOW", "Maintenance & Plumbing"),
        ("Water pressure low in 4th floor drinking water fountain.", "Plumbing", "LOW", "LOW", "Maintenance & Plumbing"),
        ("Washbasin tap handle loose in Block C rest room.", "Plumbing", "LOW", "LOW", "Maintenance & Plumbing"),
        ("Minor water seepage near window frame during rain in Room 210.", "Plumbing", "LOW", "LOW", "Maintenance & Plumbing"),
        ("Water purifier drip tray missing in staff lounge.", "Plumbing", "LOW", "LOW", "Maintenance & Plumbing"),
        ("Slight rust in water coming out of sink tap in Block B ground floor.", "Plumbing", "LOW", "LOW", "Maintenance & Plumbing"),

        # --- INFRASTRUCTURE ---
        ("Structural concrete slab collapsed on walkway near mechanical block!", "Infrastructure", "CRITICAL", "CRITICAL", "Estate & Facilities"),
        ("Elevator stuck between 3rd and 4th floor with 3 students inside Block A!", "Infrastructure", "CRITICAL", "CRITICAL", "Estate & Facilities"),
        ("Heavy plaster ceiling falling down in main lecture auditorium stage area.", "Infrastructure", "CRITICAL", "HIGH", "Estate & Facilities"),
        ("Glass panel of window high floor cracked severely and about to fall down.", "Infrastructure", "HIGH", "HIGH", "Estate & Facilities"),
        ("Main entry metal door hinge broken, door jammed half open in Block C.", "Infrastructure", "HIGH", "MEDIUM", "Estate & Facilities"),
        ("Staircase handrail loose and wobbling dangerously on 3rd floor Block B.", "Infrastructure", "HIGH", "MEDIUM", "Estate & Facilities"),
        ("Classroom desk and podium wooden legs broken in Room 305.", "Infrastructure", "MEDIUM", "MEDIUM", "Estate & Facilities"),
        ("Door lock handle stuck in room 204 unable to latch properly.", "Infrastructure", "MEDIUM", "LOW", "Estate & Facilities"),
        ("Multiple tiles cracked and lifted on hallway floor Block A ground floor.", "Infrastructure", "MEDIUM", "LOW", "Estate & Facilities"),
        ("Window latch broken allowing cold wind into Room 102.", "Infrastructure", "MEDIUM", "LOW", "Estate & Facilities"),
        ("Chairs in lecture hall 3 have loose screws and shaky armrests.", "Infrastructure", "LOW", "LOW", "Estate & Facilities"),
        ("Whiteboard frame detached on left side in Room 410.", "Infrastructure", "LOW", "LOW", "Estate & Facilities"),
        ("Door stopper missing in staff office 105.", "Infrastructure", "LOW", "LOW", "Estate & Facilities"),
        ("Curtain rod loose in seminar room 2.", "Infrastructure", "LOW", "LOW", "Estate & Facilities"),
        ("Paint peeling off wall in corridor near room 208.", "Infrastructure", "LOW", "LOW", "Estate & Facilities"),
        ("Notice board glass frame key missing in main hall.", "Infrastructure", "LOW", "LOW", "Estate & Facilities"),

        # --- IT / TECHNICAL ---
        ("Core campus network router down, entire campus disconnected from internet.", "IT / Technical", "CRITICAL", "HIGH", "IT Support Desk"),
        ("Ransomware warning popups appearing across library public workstations.", "IT / Technical", "CRITICAL", "CRITICAL", "IT Support Desk"),
        ("Server room temperature alert high due to cooling unit shutdown.", "IT / Technical", "CRITICAL", "HIGH", "IT Support Desk"),
        ("Main server switch offline disabling university portal and LMS.", "IT / Technical", "HIGH", "HIGH", "IT Support Desk"),
        ("Classroom projector in Room 302 not display output and HDMI port damaged.", "IT / Technical", "HIGH", "MEDIUM", "IT Support Desk"),
        ("The classroom projector is not working.", "IT / Technical", "MEDIUM", "MEDIUM", "IT Support Desk"),
        ("Wi-Fi network signal dropping constantly in Hostel 3 wing B.", "IT / Technical", "HIGH", "MEDIUM", "IT Support Desk"),
        ("Lab computer desktop 15 failing to boot, showing disk error screen.", "IT / Technical", "MEDIUM", "MEDIUM", "IT Support Desk"),
        ("Audio system microphone amplifier cracking loudly in auditorium.", "IT / Technical", "MEDIUM", "MEDIUM", "IT Support Desk"),
        ("LAN socket desk 8 in lab 1 damaged and not connecting.", "IT / Technical", "MEDIUM", "LOW", "IT Support Desk"),
        ("Smartboard touchscreen uncalibrated and unresponsive in Room 105.", "IT / Technical", "MEDIUM", "LOW", "IT Support Desk"),
        ("Printer in department office printing black streaks on pages.", "IT / Technical", "LOW", "LOW", "IT Support Desk"),
        ("Mouse cable damaged on computer workstation 4 in library.", "IT / Technical", "LOW", "LOW", "IT Support Desk"),
        ("Keyboard key missing on PC 12 in computer lab 3.", "IT / Technical", "LOW", "LOW", "IT Support Desk"),
        ("Monitor brightness buttons broken on lab PC 09.", "IT / Technical", "LOW", "LOW", "IT Support Desk"),
        ("VGA adapter missing in seminar hall podium.", "IT / Technical", "LOW", "LOW", "IT Support Desk"),

        # --- TRANSPORTATION ---
        ("Campus shuttle bus brake failure reported on campus ring road!", "Transportation", "CRITICAL", "CRITICAL", "Fleet & Transit"),
        ("Bus 4 breakdown near east gate blocking main exit road.", "Transportation", "HIGH", "HIGH", "Fleet & Transit"),
        ("Shuttle bus overcrowding dangerously exceeding passenger capacity.", "Transportation", "HIGH", "HIGH", "Fleet & Transit"),
        ("Morning bus route 2 arrived 45 minutes late missing first exam slot.", "Transportation", "HIGH", "MEDIUM", "Fleet & Transit"),
        ("Bus driver driving recklessly at high speed inside college zone.", "Transportation", "HIGH", "HIGH", "Fleet & Transit"),
        ("Shuttle bus air conditioning not working on route 5.", "Transportation", "MEDIUM", "MEDIUM", "Fleet & Transit"),
        ("Bus stop shelter roof leaking rainwater over seating area.", "Transportation", "MEDIUM", "LOW", "Fleet & Transit"),
        ("Schedule timings banner missing at North gate bus stop.", "Transportation", "MEDIUM", "LOW", "Fleet & Transit"),
        ("Bicycle parking rack bent and overcrowded near hostel block 2.", "Transportation", "LOW", "LOW", "Fleet & Transit"),
        ("EV charging point 2 near admin block display screen frozen.", "Transportation", "LOW", "LOW", "Fleet & Transit"),
        ("Seat fabric torn in bus seat 12 route 3.", "Transportation", "LOW", "LOW", "Fleet & Transit"),
        ("Speed bump paint worn out near gate 1 causing hard braking.", "Transportation", "LOW", "LOW", "Fleet & Transit"),

        # --- SECURITY ---
        ("Unidentified intruder spotted near girl's hostel back perimeter wall!", "Security", "CRITICAL", "CRITICAL", "Campus Security"),
        ("Physical fight and violent altercation broken out near sports ground.", "Security", "CRITICAL", "CRITICAL", "Campus Security"),
        ("Laboratory door unlocked overnight containing hazardous chemicals.", "Security", "CRITICAL", "HIGH", "Campus Security"),
        ("CCTV camera offline in high security entrance gate 2.", "Security", "HIGH", "HIGH", "Campus Security"),
        ("Suspicious unattended backpack left in central library study area.", "Security", "HIGH", "HIGH", "Campus Security"),
        ("Stolen laptop reported from hostel room 214 while door was unlocked.", "Security", "HIGH", "HIGH", "Campus Security"),
        ("Street light dark near walkway to sports complex creating security risk.", "Security", "HIGH", "MEDIUM", "Campus Security"),
        ("Biometric scanner not recognizing student IDs at hostel entry gate.", "Security", "MEDIUM", "MEDIUM", "Campus Security"),
        ("Visitor logbook missing at security checkpoint block B.", "Security", "MEDIUM", "LOW", "Campus Security"),
        ("Security guard absent from gate 3 booth during evening shift.", "Security", "HIGH", "MEDIUM", "Campus Security"),
        ("Emergency exit sign broken and unlit in auditorium corridor.", "Security", "LOW", "LOW", "Campus Security"),
        ("Keycard lock reader light blinking orange room 104.", "Security", "LOW", "LOW", "Campus Security"),
        ("Fence wire loose near north perimeter boundary.", "Security", "LOW", "LOW", "Campus Security"),

        # --- CLEANING ---
        ("Chemical spill of concentrated acid in chemistry lab 2 floor!", "Cleaning", "CRITICAL", "CRITICAL", "Sanitation & Hygiene"),
        ("Overflowing trash bins creating biohazard foul smell near cafeteria entrance.", "Cleaning", "HIGH", "HIGH", "Sanitation & Hygiene"),
        ("Washrooms in Block B ground floor extremely dirty and unhygienic.", "Cleaning", "HIGH", "MEDIUM", "Sanitation & Hygiene"),
        ("Stagnant dirty water pool near canteen attracting mosquitoes.", "Cleaning", "HIGH", "MEDIUM", "Sanitation & Hygiene"),
        ("Food waste spilled in corridor of Hostel 1 2nd floor.", "Cleaning", "MEDIUM", "MEDIUM", "Sanitation & Hygiene"),
        ("Classroom 201 whiteboard dirty and stained not erased from yesterday.", "Cleaning", "LOW", "LOW", "Sanitation & Hygiene"),
        ("Dust accumulated heavily on library top shelves and books.", "Cleaning", "LOW", "LOW", "Sanitation & Hygiene"),
        ("Corridor floor sticky due to spilled juice near vending machine.", "Cleaning", "MEDIUM", "LOW", "Sanitation & Hygiene"),
        ("Trash bin lid missing in Block A hallway.", "Cleaning", "LOW", "LOW", "Sanitation & Hygiene"),
        ("Window glass panes dusty in Room 304.", "Cleaning", "LOW", "LOW", "Sanitation & Hygiene"),
        ("Hand soap dispenser empty in rest room floor 3.", "Cleaning", "MEDIUM", "LOW", "Sanitation & Hygiene"),
        ("Paper towel dispenser empty in faculty lounge washroom.", "Cleaning", "LOW", "LOW", "Sanitation & Hygiene"),

        # --- ACADEMIC ---
        ("Exam hall question paper misprint across all copies of Math exam!", "Academic", "CRITICAL", "HIGH", "Academic Affairs"),
        ("Course portal server down preventing online assignment submission before deadline.", "Academic", "HIGH", "HIGH", "Academic Affairs"),
        ("Timetable clash between two mandatory major courses in semester 5.", "Academic", "HIGH", "MEDIUM", "Academic Affairs"),
        ("Grades missing on transcript portal for end semester exam.", "Academic", "HIGH", "MEDIUM", "Academic Affairs"),
        ("Professor absent without substitute for lecture in Room 101.", "Academic", "MEDIUM", "LOW", "Academic Affairs"),
        ("Lab manual books out of stock in university bookstore.", "Academic", "LOW", "LOW", "Academic Affairs"),
        ("Attendance record discrepancy shown in portal for CS201.", "Academic", "MEDIUM", "LOW", "Academic Affairs"),
        ("Project submission link closed early before official deadline.", "Academic", "HIGH", "MEDIUM", "Academic Affairs"),
        ("Syllabus copy mismatch between department notice board and online PDF.", "Academic", "LOW", "LOW", "Academic Affairs"),
        ("Library reference book reserved copy damaged missing pages.", "Academic", "LOW", "LOW", "Academic Affairs"),

        # --- HOSTEL ---
        ("Hostel room 304 door lock jammed, student trapped inside room!", "Hostel", "CRITICAL", "HIGH", "Hostel Management"),
        ("Severe bed bug infestation reported across entire Hostel 2 3rd floor.", "Hostel", "HIGH", "HIGH", "Hostel Management"),
        ("Washing machine in hostel laundry room leaking heavy water and shorting out.", "Hostel", "HIGH", "MEDIUM", "Hostel Management"),
        ("Hot water geyser tripping main breaker continuously in Hostel 1.", "Hostel", "HIGH", "MEDIUM", "Hostel Management"),
        ("Hostel mess food quality poor and unhygienic with undercooked items.", "Hostel", "HIGH", "HIGH", "Hostel Management"),
        ("Mattress torn and spring broken in hostel room 110.", "Hostel", "MEDIUM", "LOW", "Hostel Management"),
        ("Study room table lamp loose socket in Hostel 3 common room.", "Hostel", "LOW", "LOW", "Hostel Management"),
        ("Hostel balcony drying rack broken.", "Hostel", "LOW", "LOW", "Hostel Management"),
        ("Water cooler filter needs replacement in Hostel Block 4 floor 1.", "Hostel", "MEDIUM", "LOW", "Hostel Management"),
        ("Hostel room cup-board door handle detached.", "Hostel", "LOW", "LOW", "Hostel Management"),

        # --- OTHER ---
        ("Lost student wallet containing ID cards and keys found near main quad.", "Other", "LOW", "LOW", "Administrative Services"),
        ("Inquiry regarding summer internship certificate issuing process.", "Other", "LOW", "LOW", "Administrative Services"),
        ("Request to update emergency contact phone number in college records.", "Other", "LOW", "LOW", "Administrative Services"),
        ("General feedback regarding cafeteria seating arrangement.", "Other", "LOW", "LOW", "Administrative Services"),
        ("Lost umbrella left in seminar hall room 2.", "Other", "LOW", "LOW", "Administrative Services"),
        ("Query about library opening hours during upcoming national holiday.", "Other", "LOW", "LOW", "Administrative Services"),
        ("Request for parking permit sticker for new vehicle.", "Other", "LOW", "LOW", "Administrative Services")
    ]

    # Generate synthetic variations to enrich dataset to ~350 samples
    categories_templates = {
        "Electrical": [
            ("Light bulb blown out in {loc}.", "LOW", "LOW", "Electrical Works"),
            ("Electrical outlet sparking dangerously in {loc}!", "CRITICAL", "CRITICAL", "Electrical Works"),
            ("Short circuit causing smell of burning wire in {loc}.", "HIGH", "HIGH", "Electrical Works"),
            ("Ceiling fan in {loc} is not spinning at full speed.", "MEDIUM", "LOW", "Electrical Works"),
            ("Power supply trip in {loc} during lab session.", "HIGH", "MEDIUM", "Electrical Works"),
            ("Extension box socket burnt out in {loc}.", "MEDIUM", "MEDIUM", "Electrical Works"),
            ("Elevator light flickering inside lift shaft in {loc}.", "LOW", "LOW", "Electrical Works")
        ],
        "Plumbing": [
            ("Water leaking continuously from sink pipe in {loc}.", "HIGH", "HIGH", "Maintenance & Plumbing"),
            ("Bathroom drain clogged completely in {loc}.", "MEDIUM", "MEDIUM", "Maintenance & Plumbing"),
            ("Major water pipe leak flooding hallway in {loc}!", "CRITICAL", "HIGH", "Maintenance & Plumbing"),
            ("Drinking water tap missing handle in {loc}.", "LOW", "LOW", "Maintenance & Plumbing"),
            ("Foul sewage odor coming from drainage pipe in {loc}.", "HIGH", "HIGH", "Maintenance & Plumbing"),
            ("No water in washroom tap in {loc}.", "MEDIUM", "MEDIUM", "Maintenance & Plumbing")
        ],
        "Infrastructure": [
            ("Broken window glass with sharp edges hanging in {loc}.", "HIGH", "HIGH", "Estate & Facilities"),
            ("Door latch broken in {loc}, cannot lock room.", "MEDIUM", "LOW", "Estate & Facilities"),
            ("Ceiling plaster cracked and falling down in {loc}!", "CRITICAL", "HIGH", "Estate & Facilities"),
            ("Staircase steps chipped and dangerous in {loc}.", "MEDIUM", "MEDIUM", "Estate & Facilities"),
            ("Desk wooden leg cracked in {loc}.", "LOW", "LOW", "Estate & Facilities")
        ],
        "IT / Technical": [
            ("Classroom projector in {loc} display lamp failing.", "MEDIUM", "MEDIUM", "IT Support Desk"),
            ("Wi-Fi router offline in {loc}, zero signal.", "HIGH", "MEDIUM", "IT Support Desk"),
            ("Lab computer in {loc} keeps freezing on startup.", "MEDIUM", "MEDIUM", "IT Support Desk"),
            ("HDMI connector pin damaged on podium in {loc}.", "LOW", "LOW", "IT Support Desk"),
            ("LAN port inactive on wall near desk in {loc}.", "LOW", "LOW", "IT Support Desk")
        ],
        "Transportation": [
            ("College shuttle bus brake noise in bus route {num}.", "HIGH", "HIGH", "Fleet & Transit"),
            ("Bus schedule delayed by 1 hour at {loc}.", "MEDIUM", "MEDIUM", "Fleet & Transit"),
            ("Bus window glass loose on route {num}.", "LOW", "LOW", "Fleet & Transit")
        ],
        "Security": [
            ("Security gate lock broken at {loc}.", "HIGH", "HIGH", "Campus Security"),
            ("Suspicious stranger wandering near {loc} late night!", "CRITICAL", "CRITICAL", "Campus Security"),
            ("CCTV camera glass dirty in {loc}.", "LOW", "LOW", "Campus Security"),
            ("Unattended bag found in {loc}.", "HIGH", "HIGH", "Campus Security")
        ],
        "Cleaning": [
            ("Garbage dumping heap uncleaned in {loc}.", "HIGH", "MEDIUM", "Sanitation & Hygiene"),
            ("Floor covered with spilled coffee in {loc}.", "LOW", "LOW", "Sanitation & Hygiene"),
            ("Restroom floor slippery with water leak in {loc}.", "MEDIUM", "MEDIUM", "Sanitation & Hygiene"),
            ("Dustbins overflowing with food waste in {loc}.", "HIGH", "MEDIUM", "Sanitation & Hygiene")
        ],
        "Academic": [
            ("Course materials link broken on portal for {loc}.", "MEDIUM", "LOW", "Academic Affairs"),
            ("Lab exam equipment missing in {loc}.", "HIGH", "MEDIUM", "Academic Affairs"),
            ("Grades error in internal evaluation for {loc}.", "MEDIUM", "LOW", "Academic Affairs")
        ],
        "Hostel": [
            ("Hostel room door lock jammed in {loc}!", "CRITICAL", "HIGH", "Hostel Management"),
            ("Hot water supply turned off in {loc}.", "MEDIUM", "MEDIUM", "Hostel Management"),
            ("Window mesh screen torn in {loc}.", "LOW", "LOW", "Hostel Management")
        ],
        "Other": [
            ("General inquiry regarding campus sports facilities at {loc}.", "LOW", "LOW", "Administrative Services"),
            ("Lost property query for item lost near {loc}.", "LOW", "LOW", "Administrative Services")
        ]
    }

    locations = [
        "Block A Room 101", "Block B Classroom 204", "Hostel 3 Floor 2", "Central Library 3rd Floor",
        "Block C Lab 2", "Main Auditorium", "Hostel 1 Washroom", "Science Block Ground Floor",
        "Mechanical Workshop", "Cafeteria Rear Entry", "Admin Block 2nd Floor", "Hostel 2 Wing B",
        "Computer Lab 4", "Block D Hallway", "Sports Complex Lobby"
    ]

    all_rows = list(raw_data)

    for cat, templates in categories_templates.items():
        for i, (tmpl, prio, sev, dept) in enumerate(templates):
            for loc in locations[:6]:
                txt = tmpl.format(loc=loc, num=i+1)
                all_rows.append((txt, cat, prio, sev, dept))

    df = pd.DataFrame(all_rows, columns=["text", "category", "priority", "severity", "department"])
    df.to_csv(DATASET_PATH, index=False)
    print(f"Dataset generated and saved to {DATASET_PATH} with {len(df)} records.")
    return df

def train_and_evaluate():
    print("Loading dataset...")
    if not os.path.exists(DATASET_PATH):
        df = generate_dataset()
    else:
        df = pd.read_csv(DATASET_PATH)

    print(f"Total dataset size: {len(df)}")
    print(f"Categories count:\n{df['category'].value_counts()}")
    print(f"Priority count:\n{df['priority'].value_counts()}")

    # 1. Text Preprocessing
    def preprocess_text(t):
        t = str(t).lower()
        t = re.sub(r'[^a-z0-9\s]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    df['cleaned_text'] = df['text'].apply(preprocess_text)

    # Train/Test Split
    X_train, X_test, y_cat_train, y_cat_test, y_prio_train, y_prio_test = train_test_split(
        df['cleaned_text'], df['category'], df['priority'], test_size=0.2, random_state=42, stratify=df['category']
    )

    # 2. Vectorization
    vectorizer = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 3. Category Classifier (Logistic Regression)
    cat_model = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
    cat_model.fit(X_train_vec, y_cat_train)

    cat_pred = cat_model.predict(X_test_vec)
    cat_acc = accuracy_score(y_cat_test, cat_pred)
    cat_report = classification_report(y_cat_test, cat_pred, output_dict=True)

    print(f"\n--- Category Classifier Accuracy: {cat_acc * 100:.2f}% ---")
    print(classification_report(y_cat_test, cat_pred))

    # 4. Priority Classifier (Random Forest Classifier)
    prio_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    prio_model.fit(X_train_vec, y_prio_train)

    prio_pred = prio_model.predict(X_test_vec)
    prio_acc = accuracy_score(y_prio_test, prio_pred)
    prio_report = classification_report(y_prio_test, prio_pred, output_dict=True)

    print(f"\n--- Priority Classifier Accuracy: {prio_acc * 100:.2f}% ---")
    print(classification_report(y_prio_test, prio_pred))

    # Extract Top TF-IDF Keywords per Category
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_keywords_per_category = {}
    for i, category in enumerate(cat_model.classes_):
        # Sort coefficients
        coefs = cat_model.coef_[i]
        top_indices = coefs.argsort()[-8:][::-1]
        top_keywords_per_category[category] = list(feature_names[top_indices])

    # 5. Save Artifacts
    joblib.dump(cat_model, os.path.join("models", "category_model.pkl"))
    joblib.dump(prio_model, os.path.join("models", "priority_model.pkl"))
    joblib.dump(vectorizer, os.path.join("models", "tfidf_vectorizer.pkl"))

    metrics = {
        "dataset_size": len(df),
        "categories_count": df['category'].value_counts().to_dict(),
        "priority_count": df['priority'].value_counts().to_dict(),
        "category_accuracy": round(cat_acc * 100, 2),
        "priority_accuracy": round(prio_acc * 100, 2),
        "category_report": cat_report,
        "priority_report": prio_report,
        "top_keywords": top_keywords_per_category,
        "feature_count": len(feature_names)
    }

    with open(os.path.join("models", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nModels and metrics successfully saved to 'models/' folder.")

if __name__ == "__main__":
    train_and_evaluate()
