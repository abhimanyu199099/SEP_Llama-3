Intro to LLMs
Project: Dynamic Inference routing to reduce hallucinations in LLMs
Key Idea:
Hallucination Detection using a particular ‘signal’, use some ‘mechanism’ to reduce its effects, intervene at inference to reduce hallucination along with certain causal validation methods	
Hypothesis: 
Localization -> Certain submodules within the transformer architecture (esp. MLP neurons & Attention Heads) are causally responsible for hallucinations -> Supported by recent literature

Types of Hallucinations (non exhaustive) (taken from Yu et al.)

Knowledge enrichment hallucinations -> lower to middle layer MLPs, model lacks sufficient subject knowledge
Answer Extraction -> Upper Layer Attention heads, information is present in latent space or context window but the attention mechanism fails and the model is “distracted” and picks the wrong option		
Existing Solutions (relevant to our proposal)

HAVE (Sept,2025)((Didn’t find any associated conference) (2509.06596 )
Only focuses on attention heads, define a context sensitivity score (penalizing attention distributed over repeated tokens and rewarding heads that attend to unique context spans? ) 
Also does something called “Value Calibration” which is basically reducing the impact of sink (read useless) tokens (whitespaces etc) while calculating attention 
Lookback Lens (Oct, 2024) (EMNLP 2024) (2407.07071)  
Focus on contextual (openbook) hallucinations -> when LLMs generate misleading or unrelated content despite having access to pertinent information in the input context. 
Uses something called “Lookback Ratio” to model such hallucinations, a low LR means the model is ignoring the source text and "talking to itself" which usually leads to hallucinations
Lookback ratio -> the ratio of attention weights on the context versus newly generated tokens (for each attention head). 
employ a logistic regression classifier to predict hallucinations based on the averaged lookback ratio vector.
GAME (March, 2025) (NAACL 2025)  (2503.08963v1)
Extend the work of Lookback lens, uses lookback ratio as feature
first predicts a chunk's hallucination score; if it falls below a threshold, the chunk is regenerated using prior bias and gradient-informed edit directions to reallocate attention weights across various heads
SEP (June, 2024) (ICML 2024) (2406.15927v1)
a probabilistic approach that approximates Semantic Entropy (SE) using a simple linear classifier (probe)
During the training phase, generate many answers and calculate their Semantic Entropy
SEP trains a linear model on the hidden states of the model's activations

Mechanistic Understanding and Mitigation of Language Model Non-Factual Hallucinations (EMNLP 2024) (Yu et al.)

mechanistic interpretability study to “investigate the origins and manifestations of non-factual hallucinations in LMs”
Use logit lens and causal mediation analysis
during model inference with input x,  take the intermediate MLP outputs and attention head outputs  within a specific layer range and then project them directly to the language modeling head layer 
fine-tune the LM to optimize the a particular objective function

Our Solution:
Combination of various techniques from above solutions. It focuses on hallucinations specifically due to attention heads

i)  A dynamic gating mechanism that suppresses "Hallucination Heads" in real-time, triggered by Semantic Entropy Probes (SEPs).
ii) Causal validation steps which determine if the detected hallucination heads actually caused it or merely a correlation


Concrete Steps(Help from Gemini)  ->

Create Dataset (Study SEP in detail before doing)

i) Goal: Create a dataset of (Hidden State, Entropy Label) pairs to train the SEP classifier.

ii) Dataset: Use XSum (extreme summarization). It is prone to hallucinations.

iii) Generation Strategy:

Select 1,000 random prompts from XSum.
For each prompt, generate 5 different responses using Llama-2-7b-chat (set temperature=0.7 to encourage diversity).

iv) Labeling (The "Ground Truth"):

We need to know if the model was "confused" (High Entropy).
Method: Calculate Rouge-L overlap between the 5 generations for the same prompt.
Low Overlap (< 0.3) -> High Variance -> Label 1 (Uncertain/Hallucination likely).
High Overlap (> 0.7) -> Low Variance -> Label 0 (Confident).
Note: This approximates Semantic Entropy without needing an expensive NLI model.

Training the SEP Probe

Goal: Train a lightweight classifier to predict the label (0 or 1) from a single hidden state vector.
Extraction:
Run the model on the 1,000 prompts again (single pass).
Set output_hidden_states=True.
Extract the hidden state of the Last Token of the Prompt from the Last Layer (Layer 31 for Llama-2-7b).
Shape: You will have a matrix X of size [1000, 4096].
Training:

Use sklearn.linear_model.LogisticRegression.
Split data: 80% Train, 20% Test.
Train: clf.fit(X_train, y_train).
Save: pickle.dump(clf, 'sep_probe.pkl').
Success Metric: If Test Accuracy > 70%, the probe works. It can now detect "confusion" before the model speaks.

Modifying the Forward Pass

Goal: Intervene during inference. If SEP says "Confused," force the model to look at the context.
Location: You need to modify the LlamaAttention class in transformers/models/llama/modeling_llama.py.
The Logic (Pseudo-code for the implementation loop):
Capture State: Inside the forward pass, grab the hidden state from the previous layer.
Trigger Check (SEP):

Python
# Load your pickle probe globally
uncertainty_prob = probe.predict_proba(hidden_state)[:, 1]
threshold = 0.7 # Hyperparameter to tune
is_confused = uncertainty_prob > threshold

Calculate Signal (Lookback Ratio):

Identify context_tokens (indices 0 to N) vs generated_tokens.
Sum attention weights on context columns: attn_ctx = attn_weights[:, :, :, :N].sum(dim=-1)
Sum total attention: attn_total = attn_weights.sum(dim=-1)
LR = attn_ctx / attn_total.

Apply Mechanism (Gating):

Python
if is_confused:
     # Define Gate: Heavily penalize heads with LOW Lookback Ratio
     # If LR is low (0.1), Gate becomes small (0.1).
     # If LR is high (0.9), Gate stays high (1.0).
     gate = torch.sigmoid(lookback_ratio * alpha)

     # Apply to Value Vector
     # (This reduces the "voice" of hallucinating heads)
     attn_output = attn_output * gate
else:
     pass # Do nothing (Efficiency win)

Causal Validation (Proving it works)

Method 1: The "Knockout" (Ablation) Test
Hypothesis: If SEP flags specific heads as "bad," removing them should improve or change the output.
Action:

Find a prompt where the model hallucinates.
Run your code. See which heads got "Gated" (suppressed).
Run the model again, but manually Zero-Out those heads completely (Gate = 0).
Success: If the hallucination disappears, you have causally linked those heads to the error.

Method 2: The "Blindness" Check
Action: Force the Gate to 0 for Context-Heavy heads (High Lookback Ratio).
Result: The model should immediately start hallucinating. This proves your Lookback signal is correctly identifying the "Grounding" heads.

5) Evaluation 

Factuality:
Run 100 samples with Standard Llama.
Run 100 samples with Your Modified Llama.
Use GPT-4-Eval: "Compare Summary A and Summary B. Which is more consistent with the document?"
Latency:
Measure tokens/sec.


