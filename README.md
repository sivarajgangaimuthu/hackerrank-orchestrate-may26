# Multi-Domain Support Triage Agent

AI-powered terminal-based support triage system built for the HackerRank Orchestrate Hackathon 2026.

The agent processes support tickets across three ecosystems:

* HackerRank Support
* Claude Help Center
* Visa Support

The system uses only the provided support corpus to:

* classify support requests
* retrieve relevant support documentation
* detect sensitive/high-risk issues
* escalate unsupported or risky cases
* generate grounded responses

---

# Features

## Intelligent Ticket Classification

Classifies:

* request type
* product area
* company/domain

Supported request types:

* `product_issue`
* `feature_request`
* `bug`
* `invalid`

---

## Retrieval-Augmented Generation (RAG)

Uses:

* local markdown support corpus
* TF-IDF retrieval
* cosine similarity ranking

No external support knowledge is used.

---

## Safety & Escalation Logic

Automatically escalates:

* hacked/stolen accounts
* fraud or payment disputes
* unauthorized access issues
* legal threats
* sensitive financial/account cases

This prevents unsafe automated responses.

---

## Grounded AI Responses

Responses are generated using:

* retrieved support documents
* deterministic prompting
* Gemini API grounding

The system avoids:

* hallucinated policies
* unsupported guarantees
* fabricated troubleshooting steps

---

# Repository Structure

```text
.
├── AGENTS.md
├── README.md
├── .gitignore
├── .env.example
├── requirements.txt
│
├── code/
│   ├── main.py
│   ├── agent.py
│   ├── corpus.py
│   ├── retrieval.py
│   ├── classifier.py
│   ├── risk.py
│   └── responder.py
│
├── support_tickets/
│   ├── sample_support_tickets.csv
│   ├── support_tickets.csv
│   └── output.csv
│
├── data/
│   ├── visa/
│   ├── hackerrank/
│   └── claude/
```

---

# Architecture

## 1. Corpus Loader (`corpus.py`)

* loads markdown support documents
* extracts clean text
* prepares retrieval corpus

## 2. Retrieval Engine (`retrieval.py`)

* TF-IDF vectorization
* cosine similarity ranking
* retrieves relevant support articles

## 3. Ticket Classifier (`classifier.py`)

Classifies:

* request type
* company
* product area

Uses deterministic rule-based logic.

## 4. Risk Engine (`risk.py`)

Detects:

* fraud
* hacked accounts
* legal threats
* payment disputes
* sensitive security issues

Determines:

* `replied`
* `escalated`

## 5. Response Generator (`responder.py`)

Generates:

* grounded support response
* justification

Uses Gemini API with retrieved context.

## 6. Agent Orchestrator (`agent.py`)

Combines:

* retrieval
* classification
* risk assessment
* response generation

## 7. Pipeline Entry Point (`main.py`)

Processes:

* `support_tickets.csv`

Generates:

* `output.csv`

---

# Installation

## Clone Repository

```bash
git clone <your_repo_url>
cd hackerrank-orchestrate-may26
```

---

## Create Virtual Environment

### Windows (Git Bash)

```bash
python -m venv venv
source venv/Scripts/activate
```

### Windows (CMD)

```cmd
python -m venv venv
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Setup

Create `.env`

Example:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash-lite
```

---

# Chat Transcript Logging

AI coding assistants append conversation and ticket-processing logs to a shared external log file for hackathon AI fluency evaluation. The log is stored outside this repository:

* Windows: `%USERPROFILE%/hackerrank_orchestrate/log.txt`
* macOS/Linux: `$HOME/hackerrank_orchestrate/log.txt`

The log directory and file are created automatically if missing. Logs are append-only, existing entries are never overwritten, and the external log file must never be committed to git.

---

# Running the Project

## Run Full Pipeline

```bash
python code/main.py
```

This will:

* read support tickets
* classify requests
* retrieve relevant support documents
* detect escalation cases
* generate grounded responses
* create `support_tickets/output.csv`

---

# Output Format

Generated file:

```text
support_tickets/output.csv
```

Output columns:

* `status`
* `product_area`
* `response`
* `justification`
* `request_type`

---

# Sample Commands

## Test Retrieval

```bash
python code/retrieval.py
```

## Test Classification

```bash
python code/classifier.py
```

## Test Escalation Logic

```bash
python code/risk.py
```

## Test Response Generation

```bash
python code/responder.py
```

## Test Full Agent

```bash
python code/agent.py
```

---

# Safety Strategy

The system never:

* invents unsupported policies
* guarantees unavailable features
* exposes sensitive information
* answers high-risk security/fraud issues directly

Sensitive cases are escalated to human review.

---

# Design Goals

* deterministic behavior
* explainable reasoning
* grounded responses
* modular architecture
* safe escalation handling
* offline corpus retrieval

---

# Technologies Used

* Python
* scikit-learn
* Gemini API
* TF-IDF Retrieval
* CSV Processing
* Rule-Based Classification

---

# Hackathon

Built for:
HackerRank Orchestrate Hackathon 2026
