#!/usr/bin/env python3
"""
Enrich Clinical Classes Catalog
Categorizes all 580 conditions into 14 balanced clinical specialties and 7 anatomical body regions,
and adds patient-friendly common names and search synonyms.
"""
import json
import re

# Comprehensive mapping dictionary for all 580 conditions
# We will read existing data/processed/clinical_classes_catalog.json

with open("data/processed/clinical_classes_catalog.json", "r") as f:
    items = json.load(f)

# Specific exact overrides for accuracy
EXACT_SPECIALTY_OVERRIDES = {
    # Mental Health & Behavioral
    "Drug Abuse": "Mental Health & Behavioral",
    "Drug Abuse (Cocaine)": "Mental Health & Behavioral",
    "Drug Abuse (Methamphetamine)": "Mental Health & Behavioral",
    "Drug Abuse (Opioids)": "Mental Health & Behavioral",
    "Marijuana Abuse": "Mental Health & Behavioral",
    "Eating Disorder": "Mental Health & Behavioral",
    "Dysthymic Disorder": "Mental Health & Behavioral",
    "Impulse Control Disorder": "Mental Health & Behavioral",
    "Oppositional Disorder": "Mental Health & Behavioral",
    "Sleep Disorders": "Mental Health & Behavioral",
    "Sleep Deprivation": "Mental Health & Behavioral",
    "Conduct Disorder": "Mental Health & Behavioral",
    "Catatonia": "Mental Health & Behavioral",
    "Catalepsy": "Mental Health & Behavioral",
    "Illusions": "Mental Health & Behavioral",

    # Musculoskeletal & Rheumatology
    "Degenerative Disc Disease": "Musculoskeletal & Rheumatology",
    "Spinal Stenosis": "Musculoskeletal & Rheumatology",
    "Injury Of The Ankle": "Musculoskeletal & Rheumatology",
    "Injury To The Arm": "Musculoskeletal & Rheumatology",
    "Injury To The Leg": "Musculoskeletal & Rheumatology",
    "Injury To The Trunk": "Musculoskeletal & Rheumatology",
    "Chronic Pain Disorder": "Musculoskeletal & Rheumatology",
    "Complex Regional Pain Syndrome": "Musculoskeletal & Rheumatology",
    "Pain Disorder Affecting The Neck": "Musculoskeletal & Rheumatology",
    "Mobility Limitation": "Musculoskeletal & Rheumatology",
    "Muscular Atrophy": "Musculoskeletal & Rheumatology",
    "Myotonia": "Musculoskeletal & Rheumatology",
    "Fasciculation": "Musculoskeletal & Rheumatology",
    "Chronic Knee Pain": "Musculoskeletal & Rheumatology",
    "Carpal Tunnel Syndrome": "Musculoskeletal & Rheumatology",
    "Tietze Syndrome": "Musculoskeletal & Rheumatology",
    "Torticollis": "Musculoskeletal & Rheumatology",

    # Neurological & Spine
    "Hemiplegia": "Neurological & Spine",
    "Paraparesis": "Neurological & Spine",
    "Paresis": "Neurological & Spine",
    "Paresthesia": "Neurological & Spine",
    "Peripheral Nerve Disorder": "Neurological & Spine",
    "Persistent Vegetative State": "Neurological & Spine",
    "Phantom Limb": "Neurological & Spine",
    "Reflex, Abnormal": "Neurological & Spine",
    "Reflex, Babinski": "Neurological & Spine",
    "Quadriplegia": "Neurological & Spine",
    "Head Injury": "Neurological & Spine",
    "Memory Disorders": "Neurological & Spine",
    "Myoclonus": "Neurological & Spine",
    "Sensation Disorders": "Neurological & Spine",
    "Somatosensory Disorders": "Neurological & Spine",
    "Restless Leg Syndrome": "Neurological & Spine",
    "Horner Syndrome": "Neurological & Spine",
    "Akathisia, Drug-Induced": "Neurological & Spine",
    "Facial Pain": "Neurological & Spine",
    "Hyperalgesia": "Neurological & Spine",
    "Hyperesthesia": "Neurological & Spine",
    "Hyperkinesis": "Neurological & Spine",
    "Hypesthesia": "Neurological & Spine",
    "Hypokinesia": "Neurological & Spine",
    "Benign Paroxysmal Positional Vertical (Bppv)": "Neurological & Spine",
    "Brown-Sequard Syndrome": "Neurological & Spine",
    "Concussion": "Neurological & Spine",
    "Confusion": "Neurological & Spine",
    "Consciousness Disorders": "Neurological & Spine",

    # ENT & Oral Health
    "Dysgeusia": "ENT & Oral Health",
    "Taste Disorders": "ENT & Oral Health",
    "Olfaction Disorders": "ENT & Oral Health",
    "Halitosis": "ENT & Oral Health",
    "Hoarseness": "ENT & Oral Health",
    "Gum Disease": "ENT & Oral Health",
    "Jaw Disorder": "ENT & Oral Health",
    "Language Development Disorders": "ENT & Oral Health",
    "Language Disorders": "ENT & Oral Health",
    "Mutism": "ENT & Oral Health",
    "Speech Disorders": "ENT & Oral Health",
    "Stuttering": "ENT & Oral Health",
    "Sialoadenitis": "ENT & Oral Health",
    "Dyslexia": "ENT & Oral Health",
    "Dyslexia, Acquired": "ENT & Oral Health",
    "Gagging": "ENT & Oral Health",
    "Communication Disorders": "ENT & Oral Health",
    "Ageusia": "ENT & Oral Health",
    "Tooth Abscess": "ENT & Oral Health",
    "Tooth Disorder": "ENT & Oral Health",
    "Toothache": "ENT & Oral Health",
    "Trismus": "ENT & Oral Health",
    "Voice Disorders": "ENT & Oral Health",
    "Abscess Of The Pharynx": "ENT & Oral Health",

    # Ophthalmology (Eyes & Vision)
    "Hemianopsia": "Ophthalmology (Eyes & Vision)",
    "Iridocyclitis": "Ophthalmology (Eyes & Vision)",
    "Miosis": "Ophthalmology (Eyes & Vision)",
    "Myopia": "Ophthalmology (Eyes & Vision)",
    "Ophthalmoplegia": "Ophthalmology (Eyes & Vision)",
    "Ophthalmoplegia, Chronic Progressive External": "Ophthalmology (Eyes & Vision)",
    "Pupil Disorders": "Ophthalmology (Eyes & Vision)",
    "Scotoma": "Ophthalmology (Eyes & Vision)",
    "Amaurosis Fugax": "Ophthalmology (Eyes & Vision)",
    "Amblyopia": "Ophthalmology (Eyes & Vision)",
    "Anisocoria": "Ophthalmology (Eyes & Vision)",
    "Astigmatism": "Ophthalmology (Eyes & Vision)",
    "Vitreous Degeneration": "Ophthalmology (Eyes & Vision)",

    # Cardiovascular & Vascular
    "Premature Ventricular Contractions (Pvcs)": "Cardiovascular & Vascular",
    "Chest Pain": "Cardiovascular & Vascular",
    "Pallor": "Cardiovascular & Vascular",
    "Hypovolemia": "Cardiovascular & Vascular",
    "Cardiac Arrest": "Cardiovascular & Vascular",
    "Cardiac Output, Low": "Cardiovascular & Vascular",
    "Venous Insufficiency": "Cardiovascular & Vascular",

    # Respiratory & Thoracic
    "Snoring": "Respiratory & Thoracic",
    "Sneezing": "Respiratory & Thoracic",
    "Hypoventilation": "Respiratory & Thoracic",
    "Hypercapnia": "Respiratory & Thoracic",
    "Hyperoxia": "Respiratory & Thoracic",
    "Hypocapnia": "Respiratory & Thoracic",
    "Acute Respiratory Distress Syndrome (Ards)": "Respiratory & Thoracic",
    "Cheyne-Stokes Respiration": "Respiratory & Thoracic",
    "Common Cold": "Respiratory & Thoracic",
    "Croup": "Respiratory & Thoracic",
    "Cyanosis": "Respiratory & Thoracic",

    # Gastrointestinal & Hepatic
    "Indigestion": "Gastrointestinal & Hepatic",
    "Gallstone": "Gastrointestinal & Hepatic",
    "Rectal Disorder": "Gastrointestinal & Hepatic",
    "Hiccup": "Gastrointestinal & Hepatic",
    "Anal Fissure": "Gastrointestinal & Hepatic",
    "Choledocholithiasis": "Gastrointestinal & Hepatic",
    "Colic": "Gastrointestinal & Hepatic",

    # Renal & Urinary System
    "Temporary Or Benign Blood In Urine": "Renal & Urinary System",
    "Flank Pain": "Renal & Urinary System",
    "Albuminuria": "Renal & Urinary System",

    # Reproductive & Sexual Health
    "Prostatitis": "Reproductive & Sexual Health",
    "Vulvodynia": "Reproductive & Sexual Health",
    "Fetal Macrosomia": "Reproductive & Sexual Health",
    "Fetal Weight": "Reproductive & Sexual Health",
    "Hydrops Fetalis": "Reproductive & Sexual Health",
    "Hyperemesis Gravidarum": "Reproductive & Sexual Health",
    "Induced Abortion": "Reproductive & Sexual Health",
    "Missed Abortion": "Reproductive & Sexual Health",
    "Spontaneous Abortion": "Reproductive & Sexual Health",
    "Balanitis": "Reproductive & Sexual Health",
    "Benign Prostatic Hyperplasia (Bph)": "Reproductive & Sexual Health",
    "Threatened Pregnancy": "Reproductive & Sexual Health",
    "Vaginal Cyst": "Reproductive & Sexual Health",

    # Dermatology & Cutaneous
    "Onychomycosis": "Dermatology & Cutaneous",
    "Paronychia": "Dermatology & Cutaneous",
    "Molluscum Contagiosum": "Dermatology & Cutaneous",
    "Pilonidal Cyst": "Dermatology & Cutaneous",
    "Hirsutism": "Dermatology & Cutaneous",
    "Lipoma": "Dermatology & Cutaneous",
    "Hemangioma": "Dermatology & Cutaneous",
    "Scleroderma": "Dermatology & Cutaneous",
    "Cafe-au-Lait Spots": "Dermatology & Cutaneous",
    "Athlete'S Foot": "Dermatology & Cutaneous",

    # Hematology & Oncology
    "Sickle Cell Crisis": "Hematology & Oncology",
    "Hypergammaglobulinemia": "Hematology & Oncology",
    "White Blood Cell Disease": "Hematology & Oncology",

    # Endocrine & Metabolic
    "Overweight": "Endocrine & Metabolic",
    "Emaciation": "Endocrine & Metabolic",
    "Hyponatremia": "Endocrine & Metabolic",
    "Hot Flashes": "Endocrine & Metabolic",
    "Thinness": "Endocrine & Metabolic",
    "Weight Gain": "Endocrine & Metabolic",
    "Tetany": "Endocrine & Metabolic",
    "Virilism": "Endocrine & Metabolic",
    "Waterhouse-Friderichsen Syndrome": "Endocrine & Metabolic",
    "Cachexia": "Endocrine & Metabolic",

    # Infectious & Immunologic
    "Allergy": "Infectious & Immunologic",
    "Food Allergy": "Infectious & Immunologic",
    "Drug Reaction": "Infectious & Immunologic",
    "Insect Bite": "Infectious & Immunologic",
    "Envenomation From Spider Or Animal Bite": "Infectious & Immunologic",
    "Lymphadenitis": "Infectious & Immunologic",
    "Sarcoidosis": "Infectious & Immunologic",

    # General Constitutional
    "Asthenia": "General Constitutional",
    "Lethargy": "General Constitutional",
    "Heat Exhaustion": "General Constitutional",
    "Hypothermia": "General Constitutional",
    "Pain After An Operation": "General Constitutional",
    "Pain, Postoperative": "General Constitutional",
    "Pain, Intractable": "General Constitutional",
    "Aging, Premature": "General Constitutional",
    "Birth Weight": "General Constitutional",
    "Unconsciousness": "General Constitutional",
}

SPECIALTY_PATTERNS = [
    ("Mental Health & Behavioral", [
        "stress", "anxiety", "depress", "bipolar", "schizo", "psych", "adhd", "autis",
        "alcohol", "substance", "addict", "panic", "phobia", "delirium", "dementia",
        "bulimia", "anorexia", "insomnia", "somn", "hallucin", "personality", "asperger",
        "tic", "adjustment", "behavior", "conduct", "somatization", "conversion disorder",
        "malingering", "obsessive", "compulsive", "hypochondriasis", "neurosis", "mental"
    ]),
    ("Ophthalmology (Eyes & Vision)", [
        "eye", "vision", "ocular", "cornea", "retin", "cataract", "glaucoma", "conjunctiv",
        "astigmat", "amaurosis", "amblyopia", "anisocoria", "strabismus", "blephar", "vitreous",
        "optic", "blindness", "chalazion", "stye", "uveitis", "kerat", "diplopia", "ptosis",
        "macular", "color blindness", "trachoma", "epiphora", "night blindness", "presbyopia",
        "papilledema", "hyphaema", "scleritis", "keratitis", "hyphema"
    ]),
    ("ENT & Oral Health", [
        "ear", "hearing", "tinnitus", "otitis", "nose", "nasal", "sinus", "throat", "pharyng",
        "pharynx", "laryng", "tonsil", "tooth", "teeth", "dental", "gingiv", "oral", "mouth",
        "tongue", "vertigo", "aphasia", "aphonia", "dysphonia", "voice", "articulation",
        "epistaxis", "cleft", "auditory", "anomia", "trismus", "salivary", "ageusia", "anosmia",
        "parotitis", "labyrinthitis", "meniere", "stomatitis", "adenoid", "cheilitis", "glossitis",
        "uvulitis", "peritonsillar", "deafness", "presbycusis", "caries", "pulpitis", "periodont"
    ]),
    ("Respiratory & Thoracic", [
        "lung", "pulmon", "bronch", "asthma", "copd", "pneumon", "pleur", "cough", "croup",
        "breath", "dyspnea", "trachea", "apnea", "asphyxia", "emphysema", "hypoxia", "tuberculosis",
        "hemoptysis", "respiratory", "bronchiectasis", "atelectasis", "pneumothorax", "silicosis",
        "asbestosis", "stridor", "wheez", "hyperventilation", "pertussis", "ards", "cheyne"
    ]),
    ("Cardiovascular & Vascular", [
        "heart", "cardio", "coronary", "myocard", "arrhythm", "hyperten", "hypoten",
        "angina", "ischemi", "valve", "aortic", "mitral", "pericard", "vascular", "aneurysm",
        "arter", "vein", "venous", "thromb", "embol", "phlebitis", "endocarditis", "cardiomegaly",
        "cardiac", "tachycardi", "bradycardi", "flutter", "fibrillat", "atheroscler", "varicose",
        "edema", "cyanosis", "orthostatic", "syncope", "vasculitis", "raynaud", "claudication"
    ]),
    ("Gastrointestinal & Hepatic", [
        "gast", "stomach", "intestin", "bowel", "colon", "esophag", "reflux", "gerd",
        "peptic", "ulcer", "appendic", "colitis", "crohn", "diverticul", "constipat", "diarrhea",
        "nausea", "vomit", "liver", "hepat", "cirrhos", "jaundice", "gallbladder", "cholecyst",
        "pancrea", "hernia", "hemorrhoid", "abdomen", "celiac", "dyspepsia", "proctitis", "periton",
        "ileus", "megacolon", "ascites", "dysphagia", "melena", "hematemesis", "steatorrhea",
        "choledoch", "biliary", "flatulence", "fissure", "colic", "eructation", "malabsorption"
    ]),
    ("Neurological & Spine", [
        "brain", "neuro", "cerebr", "stroke", "tia", "epilep", "seizure", "convuls",
        "migraine", "headache", "parkinson", "sclerosis", "mening", "encephal", "neuritis",
        "neuropathy", "paralysis", "palsy", "tremor", "chorea", "athetosis", "spasm", "dystonia",
        "apraxia", "unconscious", "coma", "tetany", "ataxia", "sciatica", "myasthenia", "concussion",
        "amnesia", "agnosia", "agraphia", "bppv", "brown-sequard", "catalepsy", "catatonia",
        "confusion", "consciousness", "radiculopathy", "polyneuropathy", "neuralgia", "bell",
        "guillain", "alzheimer", "huntington", "motor neuron", "narcolepsy", "dizziness", "paraplegia"
    ]),
    ("Musculoskeletal & Rheumatology", [
        "bone", "joint", "arthr", "muscl", "myositis", "tendon", "sprain", "fracture",
        "osteopor", "osteoarth", "rheumat", "gout", "fibromyalg", "lupus", "spondyl",
        "back pain", "neck pain", "spine", "kyphosis", "scoliosis", "bursitis", "torticollis",
        "tietze", "synovitis", "ganglion", "cartilage", "rhabdomyolysis", "carpal tunnel",
        "knee pain", "hip pain", "shoulder", "strain", "myalgia", "cramp", "fasciitis",
        "polymyalgia", "ankylosing", "chondritis", "osteomyelitis", "tendinitis", "epicondylitis"
    ]),
    ("Dermatology & Cutaneous", [
        "skin", "dermat", "rash", "eczema", "psoriasis", "acne", "melanoma", "prurit",
        "itching", "urticaria", "hives", "alopecia", "nail", "hair", "erythema", "cellulitis",
        "impetigo", "shingles", "herpes zoster", "scabies", "tinea", "athlete", "callus", "wart",
        "ulcer, skin", "burn", "sebaceous", "vitiligo", "epiderm", "papule", "macule", "folliculitis",
        "blister", "boil", "carbuncle", "furuncle", "eczematous", "hyperhidrosis", "candidiasis, skin",
        "rosacea", "keloid", "corn", "gangrene, gas", "intertrigo", "pityriasis", "lichen", "cafe-au-lait"
    ]),
    ("Endocrine & Metabolic", [
        "diabet", "thyroid", "goiter", "cushing", "addison", "pituitary", "hormon",
        "obesity", "weight gain", "thinness", "hypercalc", "hypocalc", "hyperkal",
        "hypokal", "hyperglyc", "hypoglyc", "hyperlipid", "cholesterol", "acidosis", "alkalosis",
        "acromegaly", "virilism", "waterhouse", "cachexia", "metabolic", "hyperthyroid", "hypothyroid",
        "hypoglycemia", "hyperglycemia", "hyperuricemia", "dehydration"
    ]),
    ("Renal & Urinary System", [
        "kidney", "renal", "nephr", "ureter", "urethr", "bladder", "cystitis", "urinar",
        "incontin", "hematuria", "albuminuria", "proteinuria", "oliguria", "polyuria",
        "urolithiasis", "calculi", "pyelonephritis", "anuria", "dysuria", "nocturia", "glomerulonephritis",
        "hydronephrosis", "uremia", "renal failure", "nephrotic"
    ]),
    ("Reproductive & Sexual Health", [
        "pregnan", "obstet", "gynec", "uter", "ovari", "vagin", "cervic", "menstru",
        "amenorrhea", "dysmenorrhea", "pelvic", "testis", "testicular", "scrot", "penis",
        "erectile", "infertility", "vulva", "endometri", "miscarriage", "labor", "puerper",
        "prostate", "prostatic", "bph", "balanitis", "orchitis", "epididymitis", "phimosis",
        "priapism", "hydrocele", "varicocele", "ovarian", "uterine", "vaginitis", "mastitis",
        "breast", "menopause", "eclampsia", "preeclampsia", "contraception", "fibroid", "birth"
    ]),
    ("Infectious & Immunologic", [
        "infect", "virus", "viral", "bacteri", "fung", "parasit", "malaria", "sepsis",
        "measles", "mumps", "rubella", "chickenpox", "herpes", "hiv", "aids", "influenza",
        "flu", "mononucleosis", "lyme", "strep", "staph", "chlamydia", "gonorrhea", "syphilis",
        "tetanus", "cholera", "typhoid", "rabies", "amoeb", "cold", "septicemia", "botulism",
        "diphtheria", "leprosy", "toxoplasmosis", "shigellosis", "salmonellosis", "brucellosis",
        "giardiasis", "trichomoniasis", "leishmaniasis", "schistosomiasis", "hookworm", "fever"
    ]),
    ("Hematology & Oncology", [
        "anemia", "leukemia", "lymphoma", "cancer", "neoplasm", "tumor", "carcinoma",
        "platelet", "thrombocyt", "hemophil", "coagulat", "white blood cell", "polycythemia",
        "myeloma", "neutropenia", "leukopenia", "purpura", "bleeding", "hemorrhag",
        "petechiae", "ecchymosis", "splenomegaly", "lymphadenopathy", "sarcoma"
    ]),
]

def get_specialty(name):
    if name in EXACT_SPECIALTY_OVERRIDES:
        return EXACT_SPECIALTY_OVERRIDES[name]
    name_l = name.lower()
    for spec, patterns in SPECIALTY_PATTERNS:
        for p in patterns:
            if p in name_l:
                return spec
    return "General Constitutional"

SPECIALTY_TO_BODY_REGION = {
    "Ophthalmology (Eyes & Vision)": "Head & Neck",
    "ENT & Oral Health": "Head & Neck",
    "Neurological & Spine": "Head & Neck",
    "Cardiovascular & Vascular": "Chest & Thoracic",
    "Respiratory & Thoracic": "Chest & Thoracic",
    "Gastrointestinal & Hepatic": "Abdomen & Digestive",
    "Renal & Urinary System": "Pelvic & Urinary",
    "Reproductive & Sexual Health": "Pelvic & Urinary",
    "Musculoskeletal & Rheumatology": "Spine & Extremities",
    "Dermatology & Cutaneous": "Skin & Integumentary",
    "Mental Health & Behavioral": "Systemic & Whole Body",
    "Endocrine & Metabolic": "Systemic & Whole Body",
    "Infectious & Immunologic": "Systemic & Whole Body",
    "Hematology & Oncology": "Systemic & Whole Body",
    "General Constitutional": "Systemic & Whole Body",
}

# Common patient-friendly names and synonyms dictionary for top conditions and patterns
FRIENDLY_NAMES = {
    "Gastroesophageal Reflux Disease (Gerd)": "Acid Reflux / Heartburn",
    "Bronchial Asthma": "Asthma / Wheezing Episodes",
    "Acute Coronary Syndrome": "Heart Attack / Angina Chest Pain",
    "Hypertension": "High Blood Pressure",
    "Type 2 Diabetes Mellitus": "Type 2 Diabetes (High Blood Sugar)",
    "Type 1 Diabetes Mellitus": "Type 1 Diabetes (Insulin Dependent)",
    "Migraine": "Migraine Headache",
    "Migraine With Aura": "Migraine with Visual Aura",
    "Tension-Type Headache": "Tension Headache / Band-like Pressure",
    "Cluster Headache": "Cluster Headache / Severe Eye Pain",
    "Generalized Anxiety Disorder": "Generalized Anxiety / Chronic Worry",
    "Major Depressive Disorder": "Major Depression / Persistent Low Mood",
    "Panic Disorder": "Panic Attack Disorder",
    "Urinary Tract Infection (Uti)": "Urinary Tract Infection / Bladder Infection",
    "Kidney Stone": "Kidney Stones / Severe Flank Pain",
    "Acute Bronchitis": "Chest Cold / Acute Bronchitis",
    "Community-Acquired Pneumonia": "Pneumonia / Lung Infection",
    "Chronic Obstructive Pulmonary Disease (Copd)": "COPD / Chronic Emphysema & Bronchitis",
    "Influenza (Flu)": "The Flu / Influenza Virus",
    "Common Cold": "Common Cold / Runny Nose & Sneezing",
    "COVID-19": "COVID-19 Coronavirus Infection",
    "Osteoarthritis": "Wear-and-Tear Joint Arthritis",
    "Rheumatoid Arthritis": "Autoimmune Joint Arthritis",
    "Gout": "Gout Joint Attack (Big Toe / Knee)",
    "Fibromyalgia": "Fibromyalgia / Widespread Body Pain",
    "Atopic Dermatitis (Eczema)": "Eczema / Dry Itchy Skin Rash",
    "Psoriasis": "Psoriasis / Flaky Scaly Skin Plaques",
    "Herpes Zoster (Shingles)": "Shingles / Painful Blistering Nerve Rash",
    "Acne": "Acne / Breakouts & Pimples",
    "Conjunctivitis (Pink Eye)": "Pink Eye / Eye Infection & Redness",
    "Otitis Media": "Middle Ear Infection",
    "Streptococcal Pharyngitis": "Strep Throat / Bacterial Throat Infection",
    "Acute Sinusitis": "Sinus Infection / Facial Pressure",
    "Dental Caries": "Tooth Decay / Cavities",
    "Tooth Abscess": "Infected Tooth / Dental Abscess",
    "Benign Paroxysmal Positional Vertical (Bppv)": "Vertigo / Spinning Dizziness with Head Movement",
    "Carpal Tunnel Syndrome": "Carpal Tunnel / Numb Tingling Hand & Fingers",
    "Sciatica": "Sciatica / Shooting Nerve Leg Pain",
    "Chronic Knee Pain": "Chronic Knee Joint Ache",
    "Gastroenteritis": "Stomach Bug / Viral Food Poisoning",
    "Peptic Ulcer Disease": "Stomach Ulcer / Burning Mid-Abdominal Ache",
    "Gallstone": "Gallstones / Upper Right Abdominal Spasm",
    "Cholecystitis": "Gallbladder Inflammation",
    "Appendicitis": "Appendicitis / Sudden Lower Right Belly Pain",
    "Hypothyroidism": "Underactive Thyroid / Slow Metabolism & Fatigue",
    "Hyperthyroidism": "Overactive Thyroid / Rapid Heartbeat & Weight Loss",
    "Allergy": "Allergic Reaction / Histamine Flare-up",
    "Food Allergy": "Food Allergy / Digestive & Skin Sensitivity",
    "Athlete'S Foot": "Athlete's Foot / Itchy Peeling Foot Fungus",
    "Onychomycosis": "Toenail Fungus / Thick Discolored Nails",
}

def clean_symptom(s):
    return s.replace("_", " ").strip()

enriched_classes = []

for item in items:
    name = item["name"]
    spec = get_specialty(name)
    region = SPECIALTY_TO_BODY_REGION.get(spec, "Systemic & Whole Body")
    
    # Common name fallback
    common = FRIENDLY_NAMES.get(name, name)
    
    # Generate searchable synonyms
    raw_symptoms = [clean_symptom(s) for s in item.get("typical_symptoms", [])]
    synonyms = list(set([
        name.lower(),
        common.lower(),
        *raw_symptoms[:5]
    ]))
    
    enriched_classes.append({
        "id": item["id"],
        "name": item["name"],
        "common_name": common,
        "category": spec,
        "body_region": region,
        "tier": item["tier"],
        "typical_symptoms": item.get("typical_symptoms", []),
        "synonyms": synonyms
    })

# Save enriched JSON
with open("data/processed/clinical_classes_catalog.json", "w") as f:
    json.dump(enriched_classes, f, indent=2)

# Save TypeScript file
ts_content = f"""// Autogenerated Enriched Clinical Classes Catalog (All 580 Classes arranged by System & Region)
export interface ClinicalClassItem {{
  id: string;
  name: string;
  common_name: string;
  category: string;
  body_region: string;
  tier: "primary_41" | "general_50" | "all_580";
  typical_symptoms: string[];
  synonyms: string[];
}}

export const CLINICAL_CATEGORIES = [
  "All Specialties",
  "Cardiovascular & Vascular",
  "Respiratory & Thoracic",
  "Gastrointestinal & Hepatic",
  "Neurological & Spine",
  "Mental Health & Behavioral",
  "Dermatology & Cutaneous",
  "Musculoskeletal & Rheumatology",
  "Ophthalmology (Eyes & Vision)",
  "ENT & Oral Health",
  "Renal & Urinary System",
  "Reproductive & Sexual Health",
  "Endocrine & Metabolic",
  "Infectious & Immunologic",
  "Hematology & Oncology",
  "General Constitutional",
] as const;

export type ClinicalCategory = typeof CLINICAL_CATEGORIES[number];

export const BODY_REGIONS = [
  "All Regions",
  "Head & Neck",
  "Chest & Thoracic",
  "Abdomen & Digestive",
  "Pelvic & Urinary",
  "Spine & Extremities",
  "Skin & Integumentary",
  "Systemic & Whole Body",
] as const;

export type BodyRegion = typeof BODY_REGIONS[number];

export const TOP_FREQUENT_CONDITIONS = [
  "Migraine",
  "Bronchial Asthma",
  "Gastroesophageal Reflux Disease (Gerd)",
  "Acute Coronary Syndrome",
  "Hypertension",
  "Type 2 Diabetes Mellitus",
  "Urinary Tract Infection (Uti)",
  "Atopic Dermatitis (Eczema)",
  "Generalized Anxiety Disorder",
  "Osteoarthritis",
  "Acute Sinusitis",
  "Common Cold",
  "Peptic Ulcer Disease",
  "Kidney Stone",
  "Conjunctivitis (Pink Eye)",
  "Otitis Media",
  "Gout",
  "Herpes Zoster (Shingles)",
  "Fibromyalgia",
  "Hypothyroidism",
] as const;

export const CLINICAL_CLASSES: ClinicalClassItem[] = {json.dumps(enriched_classes, indent=2)};
"""

with open("frontend/lib/clinical-classes.ts", "w") as f:
    f.write(ts_content)

print(f"Successfully enriched {len(enriched_classes)} clinical classes and generated frontend/lib/clinical-classes.ts.")

# Print distribution
from collections import Counter
print("\nSpecialty Distribution:")
print(json.dumps(Counter(c["category"] for c in enriched_classes), indent=2))
print("\nBody Region Distribution:")
print(json.dumps(Counter(c["body_region"] for c in enriched_classes), indent=2))
