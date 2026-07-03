# Regulatory Analysis

*Week 1 deliverable. Written to be pasted into the notebook as markdown cells.
This is a best-effort compliance-by-design analysis, not legal advice.*

## 1. Why this section exists before a single model is trained

The single most important regulatory move in this project happened at the
**framing stage, before any code**: this system is designed as an **anonymous
age-bucket estimator for aggregate retail analytics**, not as a face-recognition
or identification system. That one decision is what keeps most of the heaviest
EU AI Act and GDPR obligations *out of scope* — so this analysis exists to make
that argument explicitly, and to show where the design choices in `data.py` and
the planned model directly implement it.

---

## 2. EU AI Act

### 2.1 Is this a "biometric categorisation system"?

The Act defines **biometric categorisation** as assigning people to categories
based on biometric data — e.g. inferring race, gender, or age from a face.
**By that definition, yes — this system is a biometric categorisation system**,
and it must be treated as one. The question is *which* obligations attach to it.

### 2.2 Prohibited practices (Article 5) — checked, not triggered

Article 5 bans AI systems that use biometric categorisation to **infer**
race, political opinions, trade union membership, religious/philosophical
beliefs, sex life, or sexual orientation, with narrow exceptions.

This is the single biggest reason the architecture keeps race/gender as
**internal, audit-only auxiliary heads that are never exposed at deployment**
(see [README](README.md#model-design)). The *deployed* system infers and
outputs **only an age bucket** — it does not infer or output race. Race is used
internally, during training/evaluation, purely to measure and correct bias in
the age prediction — arguably closer to the Act's own carve-out for
"labelling or filtering of lawfully acquired biometric datasets" for
fairness/quality purposes than to a deployed categorisation product.
**This is a load-bearing design decision, not an incidental one.**

### 2.3 High-risk classification (Annex III)

Annex III lists **biometric categorisation systems** (other than ones prohibited
outright) as high-risk when they are placed on the market as such. A high-risk
classification would require: a risk-management system, data governance
documentation, technical documentation, logging, human oversight, accuracy/
robustness/cybersecurity requirements, and conformity assessment.

**Our position:** because the deployed output is age-only (not race/gender/
identity categorisation), and the intended purpose is aggregate, anonymous
retail-footfall analytics rather than categorising *identified* individuals,
this system is argued to sit **below** the Annex III bar — but this is exactly
the kind of borderline case the Act is designed to catch, so we treat it as
**high-risk-adjacent** and voluntarily build several high-risk-grade controls
anyway, because it costs little now and removes the biggest audit risk later:

| High-risk obligation (Annex III / Art. 9-15) | What we build anyway |
|---|---|
| Risk management system | Risk register (Week 2 deliverable) |
| Data governance | `data.py` provenance columns + this document |
| Technical documentation | This README + model card |
| Accuracy & robustness testing | Fairness metrics + cross-dataset test (Week 2-3) |
| Human oversight | No automated action is taken on individuals; outputs are aggregate counts only |

### 2.4 Transparency obligations (Article 50)

Even outside the high-risk tier, the Act requires people be **informed when
subject to biometric categorisation**. In a real deployment this means visible
signage at the retail location ("this area uses AI-based visitor analytics; no
images are stored") — a concrete operational requirement flowing directly from
this analysis, not just a nice-to-have.

---

## 3. GDPR

### 3.1 Is a face image "special category" biometric data?

This is the most genuinely debated point in the whole analysis — treated
honestly rather than glossed over.

- GDPR Art. 4(14) defines biometric data as data from specific technical
  processing relating to physical characteristics **"which allow or confirm the
  unique identification of that natural person."**
- Recital 51 explicitly says photographs are **not systematically** considered
  special-category data — only when processed through specific technical means
  permitting unique identification (i.e. face *recognition*, not face
  *analysis*).
- Our system never matches a face to an identity and never stores a face
  template — it produces one categorical output (an age bucket) and discards
  the input. **On this reading, it does not process GDPR Art. 9 special-category
  biometric data.**
- **Counter-argument we do not dismiss:** regulators and courts have not fully
  settled this line for age/attribute-inference systems, and a face is
  inherently identifiable data *while it is being processed*, even briefly. A
  cautious deployment should not rely solely on the "not technically biometric
  data" argument and should still apply special-category-level safeguards as a
  matter of good practice.

### 3.2 It is personal data regardless — core principles applied

Whether or not Art. 9 applies, a face image is **personal data** for as long as
it exists in the pipeline (Art. 4(1)). The design applies GDPR's core principles
(Art. 5) directly:

| Principle | How it's implemented |
|---|---|
| **Data minimisation** | Only an age bucket is retained; the raw image/embedding is discarded immediately after inference (see workflow diagram — the 🔒 "discard image" step) |
| **Purpose limitation** | Output is used only for aggregate footfall analytics, never for individual profiling, targeting, or decisions about a specific person |
| **Storage limitation** | No image or per-person record persists after inference |
| **Lawful basis (Art. 6)** | Most plausible basis is legitimate interest (Art. 6(1)(f)) for anonymous retail analytics — this requires, and should be backed by, a documented Legitimate Interest Assessment weighing business need against customer privacy expectations |
| **Transparency (Art. 13/14)** | In-store signage / privacy notice disclosing the system's use, matching the AI Act's Art. 50 requirement above |
| **DPIA (Art. 35)** | Likely *required* in a real deployment: systematic monitoring of a publicly accessible space using a novel technology is exactly the trigger scenario Art. 35(3) and most DPA guidance flag for a mandatory Data Protection Impact Assessment |

### 3.3 The training-data licence is itself a compliance fact

UTKFace's licence is **"non-commercial research use only."** This does not
block the coursework use of this project, but it is a genuine governance point:
**if this model were ever retrained or fine-tuned for a real commercial retail
deployment, the current training data could not be reused as-is** without
re-sourcing commercially-licensed data. Documenting the licence per row in
`data.py` (the `license` column) exists specifically so this fact isn't lost
between "student project" and "someone tries to productionise it."

---

## 4. Other considerations

- **Non-discrimination:** if age-bucket analytics ever informed downstream
  decisions (e.g. different marketing/pricing by inferred age group), this
  would raise age-discrimination concerns under national consumer-protection
  and equality law — out of scope for a footfall-analytics use case, but worth
  naming explicitly as a **misuse boundary** in the model card (Week 3-4).
- **Standards:** the course references DIN SPEC 92006 / DIN's AI standardisation
  roadmap and ISO/IEC work on AI risk management — the risk-register and model
  card artefacts planned for Weeks 2-4 are structured to be compatible with
  that standards vocabulary rather than inventing a bespoke format.
- **We are not lawyers.** This is a good-faith, course-appropriate compliance-
  by-design analysis intended to demonstrate regulatory reasoning, not a
  substitute for actual legal review before any real deployment.

---

## 5. Summary: regulation → design decision traceability

The point of doing this analysis *before* training is that it is not
decorative — it already shaped concrete choices already made in the codebase:

| Regulatory driver | Concrete design decision | Where |
|---|---|---|
| AI Act Art. 5 (no race/gender inference output) | Race/gender are audit-only auxiliary heads, never deployed | `model.py` design, README |
| GDPR data minimisation | Raw image discarded immediately after inference | UI workflow diagram |
| GDPR purpose limitation | Output is an aggregate bucket, not a per-person record | System framing (README top) |
| GDPR/AI Act data governance | Every row carries `dataset_source`, `original_*`, `license` | `data.py` |
| Training-data licensing risk | `license` column tracked per dataset | `data.py` |
| AI Act transparency (Art. 50) | Documented signage/disclosure requirement for real deployment | This document |
