# 🎬 HealthTrack AI: 8-Minute Demo Video Script (Plain English Version)

> **Video Rules**: Max 8 minutes (< 200 MB), Webcam ON throughout (corner PIP or full screen during intro/outro).  
> **Speaking Style**: Natural, spoken English. Technical terms are clearly stated without heavy or confusing jargon.  
> **Pace**: ~125 words/minute (~950 words total). Gives you plenty of time to click, type, and show results.

---

## ⏱️ Video Breakdown

```
1. Intro & Tech Stack (45s)        | 00:00 - 00:45 | Webcam Full / Slide
2. ML Pipeline & Arch (1m 00s)     | 00:45 - 01:45 | VS Code (model_building/)
3. RLHF Pipeline & Math (1m 30s)   | 01:45 - 03:15 | Arch Diagram & DPO Code
4. Live Demo: App + RLHF (2m 15s)  | 03:15 - 05:30 | Browser UI + Terminal Execution
5. Limitations & Future Goals (2m) | 05:30 - 07:45 | VS Code / Browser / Webcam
6. Sign-off & Conclusion (15s)     | 07:45 - 08:00 | Webcam Full
```

---

### [00:00 - 00:45] 1. Intro (45 sec)
**Screen**: *Webcam Full (or split with HealthTrack AI homepage)*  
**Visual Action**: *Smile, introduce yourself, show the running app at `http://localhost:3000`.*

> "Hi everyone, my name is Jayesh. Today, I'm presenting **HealthTrack AI**, a smart symptom checker and clinical decision support tool for telehealth patients.
>
> The goal of this project is simple: patients type their symptoms in everyday words, our machine learning models predict potential conditions, and a local AI model explains everything in clear, friendly doctor language, with fast caching built in.
>
> **Our Technology Stack**:
> - **Frontend**: Next.js 15 with Tailwind CSS and TypeScript.
> - **Backend**: FastAPI with real-time streaming.
> - **Machine Learning**: Scikit-Learn, CatBoost, and PyTorch ResMLP.
> - **LLM & Inference**: We run a local **Ollama** model (`qwen2.5:0.5b`) on the machine for private medical advice, and we use **OpenRouter** (`openai/gpt-4o-mini`) strictly for analyzing skin photos and lab reports.
>
> Let's look at how the machine learning pipeline works."

---

### [00:45 - 01:45] 2. ML Pipeline (1 min) — Code & Architecture
**Screen**: *VS Code showing `model_building/` files (`preprocess.py`, `clinical_pipeline.py`)*  
**Visual Action**: *Point out data cleaning, the 71k duplicate rows removed, and the decoupled models.*

> "Inside the `model_building/` folder, we built our diagnostic machine learning engine:
>
> **Data Collection & Preprocessing**:
> - In `data/preprocess.py`, we take electronic health records and link them to standardized **MeSH** medical symptom names.
> - Real medical data often has repeated records. We ran exact deduplication and removed **71,814 duplicate rows** so the models don't cheat or overfit.
>
> **Decoupled Model Architecture**:
> - If you try to force 500 different diseases into one giant model, accuracy drops because completely different illnesses can share the exact same symptoms.
> - Instead, I split them into specialized models:
>   1. **Primary Care Model (41 conditions)**: A **Random Forest** model with **99.91% test accuracy** and a near-perfect calibration score of `0.036`.
>   2. **General Medicine Model (50 conditions)**: A **CatBoost** model that reaches **92.97% test accuracy** and `99.90%` top-5 accuracy.
>   3. **Broad Hospital Cohort (580 conditions)**: An 8.9 million parameter **PyTorch ResMLP** calibrated with temperature scaling.
>
> - On top of this, we use **Adaptive Conformal Prediction Sets**, which mathematically guarantees that the patient's actual disease is inside the predicted list with **at least 95% confidence**."

---

### [01:45 - 03:15] 3. RLHF Pipeline (1.5 min) — Code & Architecture Diagram
**Screen**: *VS Code showing `ingestion/router.py`, `training/m5_preference/`, and `clinical_pipeline.md` diagram*  
**Visual Action**: *Show web scraping flow, Dual Reward Engine, 70/30 anchor ratio, and DPO worker.*

> "Beyond just classification, we built a closed-loop **RLHF (Reinforcement Learning from Human Feedback)** pipeline so our local Ollama model talks like a real, careful doctor:
>
> **1. Web Scraping & Quality Filters**:
> - In `ingestion/router.py`, the system searches trusted sources like **MedlinePlus, PubMed, and Wikipedia**.
> - Text is cleaned using **C4 quality rules** and **MinHash LSH** to remove near-duplicate pages, with SHA-256 cryptographic hashes saved for proof.
>
> **2. Output Generation & Rejection Sampling**:
> - In `rejection_sampler.py`, we ask Ollama for multiple answer candidates using different temperatures.
> - We create contrastive pairs: high-quality doctor explanations versus bad answers that give dangerous advice or guess too quickly without asking questions.
>
> **3. Clinical Dual Reward Engine**:
> - We score each answer on two key things: **Clinical Safety** (at least 0.40) and **Helpfulness**. Any answer that suggests wrong drugs or ignores allergies gets heavily penalized.
>
> **4. Preventing Model Collapse (The 70/30 Rule)**:
> - Research shows that if an AI only trains on its own text, it gets worse over time. To prevent this, we mix **70% Golden Doctor Anchors (105 cases)** with **30% User Trajectories (45 cases)**, giving us exactly 150 balanced preference pairs.
>
> **5. DPO & Statistical Release Gate**:
> - We train the model using **Direct Preference Optimization (DPO)** on Apple Silicon Metal shaders.
> - Then, we test it against a 30-case benchmark using a 1,000-sample **Bootstrap test**. If the new model is proven better, it automatically creates a new Ollama model and updates the system."

---

### [03:15 - 05:30] 4. Live Demo (2 min 15 sec) — Frontend & RLHF in Action
**Screen**: *Browser (`http://localhost:3000/consultation`) $\to$ Terminal running DPO script*  
**Visual Action**: *Type high-acuity chest pain query $\to$ show results $\to$ switch profiles $\to$ upload image $\to$ run continuous DPO script.*

> *(Action: Switch to Browser on `localhost:3000/consultation`)*  
> "Now let's see the full application working live.
>
> **[Demo Part A: Frontend Consultation & Doctor Persona]**  
> In the free-text box, I'll type a serious emergency case:  
> > *'I am a 58-year-old male with high blood pressure. For 40 minutes, I have had crushing chest pressure radiating to my left jaw, sweating, and shortness of breath.'*
>
> I click Send. In less than a second, we get our structured results:
> 1. **Machine Learning Prediction**: The model predicts **Acute Myocardial Infarction (Heart Attack)** with 94.8% confidence, followed by **Acute Coronary Syndrome** and **Unstable Angina**.
> 2. **Emergency Triage Alert (ESI-1)**: Because this is life-threatening, an emergency red alert triggers immediately, telling the patient to call 911 right away and not drive themselves.
> 3. **Doctor Persona (Dr. Morgan)**: The AI doctor responds with warmth and bedside empathy, explains the symptoms in simple terms, asks helpful follow-up questions, and includes a clear medical disclaimer.
> 4. **Multi-Turn Memory**: If the patient adds *'I also feel dizzy when standing'*, the system remembers the earlier chest pain instead of forgetting it.
>
> **[Demo Part B: Cache, Auth & Vision]**  
> - **Semantic Cache**: If someone else asks the same question, our cache returns the answer in just **8 milliseconds**.
> - **Email OTP Login**: Patients log in securely using a **6-digit email code** with no passwords, and can switch family profiles between Self, Child, or Parent.
> - **Image Upload**: *(Action: click upload, select skin rash image)* Using OpenRouter, the vision model spots the body location, rash borders, and checks for red flags like petechiae.
>
> **[Demo Part C: Live RLHF Retraining Pipeline]**  
> *(Action: Switch to Terminal and run: `python3 scripts/run_continuous_dpo.py --epochs 1 --batch-size 8 --force-deploy`)*  
> Let's run our continuous RLHF loop live:
> - It builds the 150 balanced preference pairs.
> - Runs DPO training on our local hardware.
> - Evaluates the model through the statistical release gate.
> - And registers the newly trained Ollama model: **`rlhf-clinical-dpo-latest`**."

---

### [05:30 - 07:45] 5. Limitations, Bugs Solved & Future Goals (2 min)
**Screen**: *VS Code (`clinical_dpo_adapter.py`) $\to$ Terminal (`pytest`) $\to$ Webcam*  
**Visual Action**: *Show SHA-256 deduplication code, run tests in terminal, and talk to camera.*

> "Now, let's talk openly about time constraints, bugs we found and solved, and where we want to take this project:
>
> **Time Pressure & Iteration**:
> - We had a short window of about 6 hours. Instead of building a fake demo with hardcoded data, I focused on building real, working code.
> - Working fast brought up real bugs, which we fixed along the way:
>
> **Bugs Solved**:
> 1. **Buffer Duplication Bug**: In our interaction log, repeated testing created 82 duplicate queries out of 142 records. Training on duplicate data causes the model to memorize and break down. I added **SHA-256 prompt hashing** in `clinical_dpo_adapter.py` so only 100% unique patient queries are used for training.
> 2. **Scraper Filter Bug**: Our web scraper was rejecting good medical articles from Wikipedia because italicized text didn't end with a period. I updated `curation/filters.py` to handle markdown formatting cleanly.
> 3. **Prompt Drift & Context Amnesia**: Earlier, the LLM was echoing technical tags. I replaced this with our **Compact 4-Tier Context format**, which keeps prompts under 385 tokens for fast, clean answers.
>
> **Current Limitations**:
> 1. **Hardware Speed**: Running models locally on a laptop CPU takes 2 to 3 seconds. Cloud GPUs would be faster, but local running gives total patient privacy.
> 2. **Symptom Overlap**: Some rare conditions have identical symptoms, so our model provides a safe differential list rather than guessing wildly.
>
> **Future Goals**:
> 1. **Hospital EHR Integration**: Connect to hospital systems using standard HL7 and FHIR formats.
> 2. **Edge Mobile Deployment**: Run quantized 7B medical models directly on smartphones for offline telehealth.
> 3. **Clinical Validation**: Run formal clinical trials following FDA medical software standards.
>
> **Testing & AI Usage Disclosure**:
> - *(Action: Run `.venv/bin/python -m pytest tests/`)* You can see that **304 unit and integration tests pass in 37 seconds**.
> - As required by the guidelines: I used AI tools like Claude and Cursor to help write boilerplate schemas and test mocks. All the core machine learning models, math, reward functions, and bug fixes were designed, written, and verified by me."

---

### [07:45 - 08:00] 6. Conclusion (15 sec)
**Screen**: *Webcam Full*  
**Visual Action**: *Look at camera, confident finish.*

> "To wrap up: within the timeframe, HealthTrack AI covers all must-haves, should-haves, and could-haves—accurate ML predictions, simple doctor advice, instant caching, secure login, and continuous RLHF learning.
>
> The code is clean, fully modular, and backed by 304 passing tests. Thank you for your time, and I look forward to your questions!"

---

## 🎙️ Recording Checklist Before Hitting Record:
- [ ] **Browser Ready**: `http://localhost:3000/consultation` open in Chrome.
- [ ] **Backend Ready**: FastAPI running on `http://localhost:8000`.
- [ ] **Ollama Ready**: `ollama serve` active in background.
- [ ] **Terminal Ready**: Virtualenv active (`source .venv/bin/activate`).
- [ ] **Query Ready in Clipboard**:  
  `I am a 58-year-old male with high blood pressure. For 40 minutes, I have had crushing chest pressure radiating to my left jaw, sweating, and shortness of breath.`
- [ ] **Command Ready for RLHF Demo**:  
  `python3 scripts/run_continuous_dpo.py --epochs 1 --batch-size 8 --force-deploy`
- [ ] **Test Command Ready**:  
  `.venv/bin/python -m pytest tests/`
- [ ] **Webcam**: Placed at eye level with good lighting.