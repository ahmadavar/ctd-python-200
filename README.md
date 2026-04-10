# Python 200 — Cloud & AI

11-week course at Code the Dream covering production data engineering: statistical analysis, the full machine learning stack, large language models, retrieval-augmented generation, AI agents, and cloud infrastructure on Microsoft Azure.

---

## Curriculum

### Week 1 — Data Analysis & Pipelines
**Folder:** `week-01-analysis-pipelines/`

Review of core data tools in a new analytical context, plus an introduction to reproducible pipeline orchestration.

| Topic | Tools |
|---|---|
| Pandas, NumPy, Matplotlib review | pandas, numpy, matplotlib |
| Descriptive statistics | mean, median, variance, distributions |
| Hypothesis testing | scipy.stats (t-tests, p-values) |
| Correlation | Pearson correlation, seaborn heatmaps |
| Data pipelines | Prefect (@task, @flow, get_run_logger) |

**Assignment:** Warmup exercises covering all topics + mini-project building a Prefect pipeline on the World Happiness dataset (2015–2024), including multi-year EDA, hypothesis testing, and a Bonferroni-corrected correlation analysis.

---

### Week 2 — Machine Learning: Regression
**Folder:** `week-02-linear-regression/`

Introduction to supervised learning and the scikit-learn API through linear regression.

| Topic | Tools |
|---|---|
| ML landscape overview | supervised vs. unsupervised, regression vs. classification |
| scikit-learn API | create → fit → predict pattern |
| Linear regression | LinearRegression, train/test splits |
| Model evaluation | MSE, RMSE, R², predicted vs. actual plots |
| Multiple regression | numeric + binary features, coefficient interpretation |

**Assignment:** Warmup on scikit-learn fundamentals + mini-project predicting student math performance (UCI dataset) using demographic and behavioral features — without using prior grade data.

---

### Week 3 — Machine Learning: Classification
**Folder:** `week-03-classification/`

Classification algorithms, data preprocessing for ML, and robust model evaluation.

| Topic | Tools |
|---|---|
| Data preprocessing | StandardScaler, train/test split, PCA |
| k-Nearest Neighbors | KNeighborsClassifier, cross-validation |
| Classifier evaluation | accuracy, precision, recall, F1, confusion matrix |
| Decision Trees & Random Forests | DecisionTreeClassifier, RandomForestClassifier, feature importances |
| Logistic Regression | regularization (C parameter), liblinear solver |
| scikit-learn Pipelines | Pipeline, chaining scaler + classifier |

**Assignment:** Warmup on all preprocessing and classification tools + mini-project building a spam classifier on the Spambase dataset — comparing 5 classifiers, cross-validating, and packaging the best model in a sklearn Pipeline.

---

### Week 4 — Deep Learning
**Folder:** `week-04-deep-learning/`

Neural networks and computer vision using PyTorch. Assignments run in Kaggle notebooks (GPU required).

| Topic | Tools |
|---|---|
| Neural network fundamentals | forward pass, loss, backpropagation |
| PyTorch tensors | torch.Tensor, GPU operations, device management |
| Pretrained CNNs | torchvision.models (ResNet18, MobileNetV3, EfficientNet-B0) |
| Image classification | ImageNet inference, preprocessing transforms |
| Transfer learning | feature extraction, fine-tuning classification heads |

**Assignment format:** Kaggle notebooks (`.ipynb`). Warmup covers PyTorch tensor operations and single-model inference. Mini-project: multi-model comparison (ResNet18 vs MobileNetV3 vs EfficientNet-B0) on the Intel Image Classification dataset — evaluating accuracy, confidence, and inference latency.

---

### Week 5 — Large Language Models & Prompt Engineering
**Folder:** `week-05-llms-prompting/`

How LLMs work and how to use them effectively through APIs.

| Topic | Tools |
|---|---|
| LLM fundamentals | tokenization, embeddings, attention (conceptual) |
| OpenAI Chat Completions API | gpt-4o-mini, temperature, max_tokens, system messages |
| Conversation memory | stateless API + manual history management |
| Prompt engineering | zero-shot, one-shot, few-shot, chain-of-thought, structured output (JSON), delimiters |
| Moderation | OpenAI moderation endpoint |
| Local models | Ollama (qwen3:0.6b) |
| AI ethics | bias, energy use, responsible deployment |

**Assignment:** Warmup covering all prompt engineering techniques + mini-project building a job application assistant chatbot with bullet point rewriting, cover letter generation, and conversation memory.

---

### Week 6 — Retrieval-Augmented Generation (RAG)
**Folder:** `week-06-rag/`

Augmenting LLMs with external knowledge through semantic search and production RAG frameworks.

| Topic | Tools |
|---|---|
| LLM augmentation strategies | prompt injection vs. fine-tuning vs. RAG |
| Keyword RAG | token overlap retrieval from scratch |
| Semantic RAG | vector embeddings, cosine similarity, FAISS |
| Persistent vector storage | pgvector (Docker) |
| Production RAG | LlamaIndex (VectorStoreIndex, query engine, evaluators) |
| RAG evaluation | FaithfulnessEvaluator, RelevancyEvaluator |

**Assignment:** Warmup on RAG concepts, keyword retrieval, and LlamaIndex fundamentals + mini-project building a full RAG-powered Q&A assistant over Groundwork Coffee Co. internal documents.

---

### Week 7 — AI Agents
**Folder:** `week-07-ai-agents/`

Building autonomous systems that plan and execute multi-step tasks.

| Topic | Tools |
|---|---|
| Agent concepts | ReAct loop (Reason + Act), tool-based vs. code-based agents |
| Custom agents from scratch | defining tools, orchestration logic |
| ETL agent | extract/transform/load with agent orchestration |
| Production framework | smolagents (HuggingFace) |

**Assignment:** Coming soon.

---

### Week 8 — Cloud Computing Introduction
**Folder:** `week-08-cloud-intro/`

Cloud fundamentals and hands-on orientation in Microsoft Azure.

| Topic | Tools |
|---|---|
| Cloud concepts | IaaS / PaaS / SaaS, vertical vs. horizontal scaling |
| Azure fundamentals | Portal, resource groups, subscriptions |
| Azure CLI | az commands, Cloud Shell, persistent storage |
| SSH key management | key generation, secure remote access |
| Cost analysis | Azure Pricing Calculator |

**Assignment:** Written warmup (conceptual questions) + video project demonstrating Azure portal navigation and a cost analysis comparing lightweight vs. GPU compute scenarios. Short Python script run in Cloud Shell.

**Note:** Week 8 assignments are in Markdown (`.md`) rather than Python.

---

### Week 9 — Data in the Cloud
**Folder:** `week-09-cloud-storage/`

Moving data engineering workflows into the cloud using Azure Blob Storage.

| Topic | Tools |
|---|---|
| Azure authentication | DefaultAzureCredential, credential chain |
| Azure Blob Storage | azure-storage-blob SDK (CRUD operations) |
| Cloud ETL — Extract & Load | REST API → Blob Storage pipeline |

**Assignment:** Coming soon.

---

### Week 10 — LLMs in Pipelines
**Folder:** `week-10-llms-pipelines/`

Using LLMs as a transform step inside cloud data pipelines.

| Topic | Tools |
|---|---|
| LLM-assisted transformation | classification, extraction, summarization at scale |
| Blob Storage integration | read from Blob → LLM enrichment → write back |
| Azure OpenAI | migrating from OpenAI API to Azure OpenAI for production |

**Assignment:** Coming soon.

---

### Week 11 — End-to-End Cloud ETL (Capstone)
**Folder:** `week-11-cloud-etl/`

Final project connecting all cloud and AI skills into a production-grade orchestrated pipeline.

| Topic | Tools |
|---|---|
| Full ETL pipeline | Prefect flow with extract (REST API) + transform (LLM) + load (Blob Storage) |
| Production patterns | retries, raise-on-failure, structured logging |
| Pipeline monitoring | Prefect UI, run states, task logs, failure traces |

**Assignment:** Coming soon.

---

## Folder Structure

```
python-200/
├── week-01-analysis-pipelines/
│   ├── warmup_01.py
│   ├── prefect_warmup.py
│   ├── project_01.py
│   └── outputs/
├── week-02-linear-regression/
│   ├── warmup_02.py
│   ├── project_02.py
│   └── outputs/
├── week-03-classification/
│   ├── warmup_03.py
│   ├── project_03.py
│   └── outputs/
├── week-04-deep-learning/          ← Kaggle notebooks (.ipynb)
│   ├── warmup_04.ipynb
│   ├── project_04.ipynb
│   └── outputs/
├── week-05-llms-prompting/
│   ├── warmup_05.py
│   ├── project_05.py
│   └── outputs/
├── week-06-rag/
│   ├── warmup_06.py
│   ├── project_06.py
│   └── outputs/
├── week-07-ai-agents/              ← Assignment TBD
│   └── outputs/
├── week-08-cloud-intro/            ← Markdown + short Python script
│   ├── warmup_08.md
│   ├── project_08.md
│   ├── project_08.py
│   └── outputs/
├── week-09-cloud-storage/          ← Assignment TBD
│   └── outputs/
├── week-10-llms-pipelines/         ← Assignment TBD
│   └── outputs/
└── week-11-cloud-etl/              ← Assignment TBD (capstone)
    └── outputs/
```

## Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Core data stack
pip install pandas numpy matplotlib seaborn scipy

# Pipelines
pip install prefect

# Machine learning
pip install scikit-learn

# LLMs and RAG
pip install openai python-dotenv llama-index-core llama-index-embeddings-openai

# Azure
pip install azure-storage-blob azure-identity

# Local models
# Install Ollama separately: https://ollama.com
```

API keys go in a `.env` file — never committed to the repo.
