'''
Common utils and experiment params
'''

MODEL_NAME = "Llama-2-7b-chat"
NLI_MODEL = "microsoft/deberta-v2-xlarge-mnli"

# --- QA Configuration (OATML paper, Llama-2-7B short-form) ---
QA_DATASETS = ["squad", "trivia_qa", "nq", "bioasq"]
NUM_SAMPLES_QA = 2000
NUM_GENERATIONS_QA = 10          # high-temp generations for SE
TEMPERATURE_HIGH = 1.0           # high-temp for SE generations
TEMPERATURE_LOW = 0.1            # low-temp for "most likely" + latent extraction
NUM_FEW_SHOT = 5
MAX_NEW_TOKENS = 50
SEED_QA = 20                     # OATML random_seed=20
BRIEF_PROMPT = "Answer the following question as briefly as possible.\n"
USE_CONTEXT = False              # short-form: no context prefix
CONDITION_ON_QUESTION = True     # prepend question to generations for NLI
STRICT_ENTAILMENT = True         # both NLI directions must be entailment (class 2)
OUTPUT_BASE = "output"           # output/{dataset_name}/

# --- XSum Configuration (long-form summarization, Lookback Gate experiment) ---
# Lookback Ratio requires multiple generated tokens to observe attention drift;
# XSum is the correct testbed (QA answers are too short for drift to manifest).
XSUM_DATASETS = ["xsum"]
XSUM_MAX_NEW_TOKENS = 100        # summaries are much longer than QA answers
XSUM_NUM_SAMPLES = 1000
XSUM_NUM_GENERATIONS = 5         # stochastic summaries for SE computation
XSUM_TEMPERATURE_HIGH = 1.0
XSUM_TEMPERATURE_LOW = 0.1
XSUM_NUM_FEW_SHOT = 3            # fewer examples to avoid very long prompts
XSUM_BRIEF_PROMPT = "Summarize the following article in one sentence.\n"
XSUM_USE_CONTEXT = True
XSUM_ACC_THRESHOLD = 0.2         # ROUGE-L / F1 threshold: summaries score ~0.2-0.4

# All datasets (QA + summarization) — used for --dataset choices
ALL_DATASETS = QA_DATASETS + XSUM_DATASETS

# --- Legacy XSum (Llama-3 pipeline, kept for compatibility) ---
NUM_SAMPLES_XSUM = 1000
NUM_GENERATIONS_XSUM = 5
TEMPERATURE = 0.7
SEED = 42