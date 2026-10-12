\# JEVTUNE — Decision Model Fine-Tuning



Fine-tuning a Qwen-based language model for structured customer-service decision prediction using \*\*Unsloth, LoRA/QLoRA, and Hugging Face Transformers\*\*.



The model predicts multiple structured decision attributes from customer-service inputs:



\- \*\*Team\*\* — `billing`, `technical`, `sales`

\- \*\*Refund\*\* — `true`, `false`

\- \*\*Urgency\*\* — `0`, `1`, `2`



\## Architecture



```text

Customer Service Input

&#x20;       │

&#x20;       ▼

&#x20;  Qwen Model

&#x20;       │

&#x20;       ▼

&#x20;LoRA / QLoRA Fine-tuning

&#x20;       │

&#x20;       ▼

&#x20;Structured Decision Output

&#x20;  ├── Team

&#x20;  ├── Refund

&#x20;  └── Urgency







Tech Stack

\- Python

\- PyTorch

\- Hugging Face Transformers

\- Unsloth

\- PEFT / LoRA

\- Hugging Face Datasets

\- CUDA

\- NVIDIA RTX 5060 Ti 16GB







jevtune/

│

├── train.py                 # Fine-tuning pipeline

├── predict.py               # Model inference and evaluation

├── dataset.ipynb            # Dataset exploration / preparation

├── typed\_decisions.csv      # Decision dataset

├── .gitignore

└── README.md





The trained model weights are intentionally not stored in this GitHub repository because model files are large.





The project uses the LocalLLaMA/typed-decisions dataset with the customer\_service configuration.

The dataset is transformed into training-ready decision records before fine-tuning.





Training Results

Initial holdout evaluation produced:

Metric	Result

Accuracy	77.33%

ECE	0.0468

Loss	1.0236

Record Accuracy	33.33%





The training run completed successfully in approximately 7.7 minutes on an RTX 5060 Ti 16GB.

Note: The holdout set contained only 30 examples, so these results should not be treated as a statistically robust benchmark.



A separate inference evaluation produced:

Metric	Result

Accuracy	53.25%

ECE	0.2012

Loss	1.4281

Record Accuracy	12.75%

