from unsloth import FastDecisionModel
from datasets import load_dataset

model, tokenizer = FastDecisionModel.from_pretrained(
    "qwen-decisions", load_in_4bit=True
)

FastDecisionModel.for_inference(model)
answers = FastDecisionModel.predict(
    model,
    tokenizer,
    "Hi, I was charged twice for invoice #4411. Please refund the duplicate today.",
    {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "invoices, payments, refunds",
                "technical": "bugs, outages, errors",
                "sales": "pricing, new plans",
            },
        },
        "refund": {
            "type": "noul",
            "instructions": "Does the customer ask for a refund?",
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["not urgent", "soon", "today"],
        },
    },
)
for name, result in answers.items():
    print(
        name,
        result["answer"],
        {k: round(v, 3) for k, v in result["probabilities"].items()},
    )


test = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
test_items, _ = FastDecisionModel.build_dataset(test, tokenizer, model)
print(FastDecisionModel.evaluate(model, tokenizer, test_items))
