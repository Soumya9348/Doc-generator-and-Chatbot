# Teaching Agent — Final Implementation Plan

> All open questions resolved. Ready to build.

---

## Knowledge Sourcing — Complete Map

### Modules 1-3: Industry Knowledge (Official Specs)

**I generate this content** from official, public specifications:

| Module | Content | Source | URL |
|--------|---------|--------|-----|
| 1. EV Ecosystem | CPO, MSP/EMSP, CPMS, roaming, money flow | OCPI 2.2.1 spec + EU AFIR | evroaming.org, github.com/ocpi/ocpi |
| 2. Protocols | OCPP (CS↔CSMS), OCPI (CPO↔MSP), versions | OCA + EVRoaming | openchargealliance.org, evroaming.org |
| 3. Hardware | Location→EVSE→Connector→Session hierarchy | OCPI 2.2.1 Locations module + IEC 62196 | github.com/ocpi/ocpi/mod_locations |

### Module 4: Source Systems (Auto-Extracted from EUH Docs)

**No separate KT script needed.** The source-to-table mapping is already documented in the EUH layer docs inside `copilot_knowledge_chunks`:

```python
# Auto-extract from existing knowledge base
source_docs = run_sql("""
    SELECT source_name, section_header, chunk_text 
    FROM emobility-uc-dev.sandbox-emobility.copilot_knowledge_chunks
    WHERE data_layer = 'euh' 
      AND section_type IN ('source_overview', 'table_details', 'notebook_purpose')
    ORDER BY source_name
""")
# This gives us: which sources (driivz, spirii, greenlots, ecomovement, cxm)
# feed into which tables (charger_location, charger_evse, etc.)
```

Known mapping (from user input):
- **All CPMS sources** (Driivz, EcoMovement, Spirii, CXM, GREENLOTS) → `charger_location`, `charger_evse`, `charger_connector`
- **Some sources** → `charger_session` (identifiable from EUH layer docs)

### Module 5: Data Model (Auto-Extracted from Unity Catalog)

```python
# Auto-extract table schemas
catalog_path = "emobility-uc-dev.euh-emobility"
tables = ["charger_location", "charger_evse", 
          "charger_connector", "charger_session"]

for t in tables:
    spark.sql(f"DESCRIBE TABLE {catalog_path}.{t}")
    spark.sql(f"SELECT COUNT(*) FROM {catalog_path}.{t}")
    spark.sql(f"SELECT * FROM {catalog_path}.{t} LIMIT 3")
```

### Modules 6-7: Pipeline & Gotchas (Existing Knowledge Chunks)

Already in `copilot_knowledge_chunks` table:
- `section_type = 'data_layer'` → Landing/Raw/EUH descriptions
- `section_type = 'transformation_steps'` → per-source transformations
- `section_type = 'deduplication_logic'` → dedup patterns
- `section_type = 'business_rules'` → field derivation rules

---

## Architecture

### State Machine

```
WELCOME → CURRICULUM → [TEACHING → DOUBT_CHECK → QUIZ → QUIZ_REVIEW] × 7 → FINAL_TEST → RESULTS
```

### Onboarding Session (stored in orchestrator per conversation)

```python
class OnboardingSession:
    user_name = ""
    user_role = ""           # "engineer" | "analyst" | "business"
    state = "welcome"        # current state machine position
    current_module = 0       # 0-6 (index into ONBOARDING_MODULES)
    module_scores = [None]*7 # quiz score per module (e.g., [3, 2, None, ...])
    quiz_progress = 0        # current question index within quiz
    quiz_answers = []        # answers for current quiz/test
    final_answers = []       # answers for final test
    final_score = 0.0        # final test percentage
    final_test_start = None  # timestamp for 20-min timer
    history = []             # conversation with answers for adaptive teaching
```

---

## Module Content & Quiz Bank

### Module 1: EV Charging Ecosystem

**Teaching content** (sourced from OCPI 2.2.1 Section 1, EU AFIR Article 2):
- CPO: owns/operates physical charging stations. Our CPOs: GREENLOTS, Driivz, Spirii, Enovos
- MSP/EMSP: customer-facing digital service (app, RFID card). Our MSP: CXM
- CPMS: software CPOs use to monitor chargers remotely
- Roaming: OCPI enables drivers to charge across CPO networks via one MSP app
- Money flow: Driver → MSP → Roaming Hub → CPO

**Quiz** (3 MCQs):
1. "A company that owns physical charging stations is called:" → B) CPO
2. "CXM in our platform serves as:" → C) Mobility Service Provider
3. "When a driver uses one app at any charger, this is enabled by:" → B) MSP + OCPI roaming

### Module 2: Communication Protocols

**Teaching content** (sourced from OCA, EVRoaming Foundation):
- OCPP: Charger hardware ↔ CPMS. Versions: 1.6 (widely deployed), 2.0.1 (latest)
- OCPI: CPO systems ↔ MSP systems. Version: 2.2.1. Modules: Locations, Sessions, Tariffs, Tokens, Commands
- Key difference: OCPP = within one company's infrastructure; OCPI = between companies

**Quiz**:
1. "OCPP handles communication between:" → A) Charger hardware and CPMS
2. "OCPI enables data exchange between:" → C) CPO systems and MSP systems
3. "Which protocol would Driivz use to send session data to CXM?" → B) OCPI

### Module 3: Hardware Hierarchy

**Teaching content** (sourced from OCPI 2.2.1 Locations module):
- Location (1) → EVSE (many) → Connector (many)
- Location: physical site (address, coordinates, operator, country)
- EVSE: one charging point with unique `evse_id` (per eMI3 standard), has `status` (AVAILABLE, CHARGING, OUT_OF_ORDER)
- Connector: physical plug (Type 2, CCS2, CHAdeMO), has `power_type` (AC/DC), `max_voltage`, `max_amperage`
- Session: one charging event linked to a connector (start/end time, energy_kwh, cost)

**Quiz**:
1. "How many connectors can one EVSE have?" → C) Multiple
2. "The EVSE status 'CHARGING' means:" → A) A vehicle is actively charging
3. "A CCS2 connector supports:" → D) DC fast charging

### Module 4: Our Source Systems

**Teaching content** (auto-extracted from `copilot_knowledge_chunks` WHERE `data_layer='euh'`):
- All CPMS sources feed into `charger_location`, `charger_evse`, `charger_connector`
- Some sources also feed `charger_session`
- Detailed mapping pulled from EUH docs at runtime

**Quiz**:
1. "Which of these is NOT a CPO source in our platform?" → D) CXM (CXM is MSP)
2. "EcoMovement provides:" → B) Aggregated charger location data
3. "All CPMS sources feed into:" → A) charger_location, charger_evse, charger_connector

### Module 5: Our Data Model

**Teaching content** (auto-extracted from `DESCRIBE TABLE emobility-uc-dev.euh-emobility.charger_*`):
- Actual column names, types, comments from Unity Catalog
- Row counts, sample data
- Relationships: location_id links location→evse, evse_id links evse→connector

**Quiz**:
1. "Which table stores physical site addresses?" → A) charger_location
2. "The charger_evse table is linked to charger_location by:" → B) location_id
3. "Charging event data (start time, energy, cost) is stored in:" → D) charger_session

### Module 6: Pipeline Architecture

**Teaching content** (from `copilot_knowledge_chunks` WHERE `section_type='data_layer'`):
- Landing: Raw API ingestion (JSON/CSV), no transformations
- Raw: Cleaned, typed, deduplicated, schema enforcement
- EUH: Business logic, cross-source harmonization, unified model
- Each source has its own pipeline through all 3 layers

**Quiz**:
1. "In which layer is raw JSON from APIs first stored?" → A) Landing
2. "Deduplication happens in which layer?" → B) Raw
3. "Business rules and cross-source harmonization happen in:" → C) EUH

### Module 7: Common Gotchas

**Teaching content** (from knowledge chunks + common patterns):
- Duplicate EVSEs across sources → dedup logic
- Connector status mapping inconsistencies
- Composite primary keys in session table
- Timezone handling in session timestamps
- NULL handling in power_kw calculations

**Quiz**:
1. "Why might the same EVSE appear twice in our data?" → B) Multiple sources report the same charger
2. "When power_kw is NULL, we should:" → C) Check source-specific derivation logic
3. "Session timestamps should always be interpreted as:" → A) UTC

---

## Final Comprehensive Test

**10 questions, 20-minute timer, 80% pass mark (8/10 minimum)**

| # | Question | Type | From Module |
|---|----------|------|-------------|
| 1 | "What is the role of a CPO vs an MSP?" | Short answer | 1 |
| 2 | "OCPP handles communication between __ and __" | MCQ | 2 |
| 3 | "Draw the hierarchy: Location → ? → ? → Session" | Short answer | 3 |
| 4 | "Which source systems feed charger_session?" | MCQ | 4 |
| 5 | "Name 3 columns in the charger_evse table" | Short answer | 5 |
| 6 | "What transformations happen in the EUH layer?" | MCQ | 6 |
| 7 | "How does our platform handle duplicate EVSEs?" | MCQ | 7 |
| 8 | "CXM is a __. GREENLOTS is a __." | MCQ | 1, 4 |
| 9 | "A driver starts a session. Trace the data flow from charger to our database." | Short answer | 2, 6 |
| 10 | "An EVSE has status OUT_OF_ORDER. Which table shows this?" | MCQ | 3, 5 |

**Grading**: MCQs auto-graded. Short answers graded by LLM with rubric:
```
Score 1: Answer shows understanding of the concept with key terms present
Score 0: Answer is wrong, vague, or missing key terms
```

---

## Certificate

**No Delta table logging.** Instead, generate a **downloadable certificate image** the newcomer can share.

### Certificate Design

```
┌──────────────────────────────────────────────────────┐
│  ┌──────────────────────────────────────────────┐    │
│  │          🎓 Certificate of Completion         │    │
│  │                                               │    │
│  │  This certifies that                          │    │
│  │                                               │    │
│  │         ✦ Soumya Sourav Behera ✦              │    │
│  │                                               │    │
│  │  has successfully completed the               │    │
│  │  eMobility Data Platform Onboarding           │    │
│  │                                               │    │
│  │  Score: 9/10 (90%) — PASSED                   │    │
│  │  Date: May 10, 2026                           │    │
│  │                                               │    │
│  │  Modules Completed:                           │    │
│  │  ✅ EV Ecosystem  ✅ Protocols  ✅ Hardware    │    │
│  │  ✅ Sources  ✅ Data Model  ✅ Pipeline        │    │
│  │  ✅ Best Practices                            │    │
│  │                                               │    │
│  │  Completion Code: EMO-2026-A7F3               │    │
│  │  ─────────────────────────────────────────    │    │
│  │  ⚡ eMobility Copilot • 2GOEnergy             │    │
│  └──────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

### Implementation

- Generated client-side using HTML Canvas → downloadable as PNG
- Includes a unique **completion code** (hash of name + date + score) for verification
- "Download Certificate" button + "Copy Completion Code" button
- Styled with gradient border, premium typography

---

## File Changes Summary

| File | Action | What |
|------|--------|------|
| `copilot_app/teaching_agent.py` | **NEW** | OnboardingSession class, state machine, teaching/quiz/test handlers, ONBOARDING_MODULES constant with all content + quizzes |
| `copilot_app/orchestrator.py` | **MODIFY** | Add ONBOARDING intent, route to teaching_agent, store answers in history, expand history to 10 turns |
| `copilot_app/static/index.html` | **MODIFY** | Onboarding chip, curriculum card, quiz options (A/B/C/D buttons), progress bar, timer, certificate canvas, flashcard flip |
| `copilot_app/app.py` | **MODIFY** | Add `/api/onboarding/reset` endpoint |
| `notebooks/assemble_teaching_content.py` | **NEW** | Auto-extract schemas from `emobility-uc-dev.euh-emobility.charger_*` + pull EUH source docs from knowledge chunks |

---

## Build Order

1. **`teaching_agent.py`** — Core module with all content, state machine, quiz logic
2. **`orchestrator.py`** — Wire up ONBOARDING intent + routing
3. **`index.html`** — All UI components (curriculum, quiz, certificate, etc.)
4. **`app.py`** — Reset endpoint
5. **`assemble_teaching_content.py`** — Schema extraction notebook
6. **Test** — Full flow walkthrough

---

## Verification Plan

| Step | Test | Pass Criteria |
|------|------|--------------|
| 1 | Click "🎓 Start Onboarding" | Welcome message, asks name |
| 2 | Type name, select role | Curriculum displayed with 7 modules |
| 3 | Start Module 1 | Teaches CPO/MSP with source citations |
| 4 | Ask "what is MSP" twice | **Different** explanation each time |
| 5 | Click "Show example" | Pulls live data from Genie |
| 6 | Say "I'm ready" | 3 MCQ questions presented |
| 7 | Answer quiz | Instant ✅/❌ + explanation per question |
| 8 | Complete all 7 modules | Final test unlocked |
| 9 | Take final test | 20-min timer, 10 questions |
| 10 | Score ≥80% | Certificate generated with download button |
| 11 | Score <80% | Shows weak modules + retry option |
| 12 | Refresh mid-module | Resumes from last position |
