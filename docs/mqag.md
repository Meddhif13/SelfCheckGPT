# MQAG Pipeline

The Multiple-choice Question Answering and Generation (MQAG) metric follows the original paper's implementation:

## Pipeline Overview

1. Stage G.1: Question + Answer Generation
   - Input: Text passage
   - Output: Question and answer pair, separated by a special token
   - Model: T5 fine-tuned on RACE/SQuAD datasets

2. Stage G.2: Distractor Generation
   - Input: Question + Answer + Context
   - Output: 3 distractor options
   - Model: T5 fine-tuned on RACE dataset

3. Multiple-Choice QA System
   - Takes questions with 4 options (correct answer + 3 distractors)
   - Uses Longformer model for answering
   - Returns probability distribution over options

## Implementation Details

- Uses sampling instead of beam search for more diverse questions
- max_new_tokens=128 for both question and distractor generation
- temperature=0.7 for controlled randomness
- Proper handling of special tokens (pad, eos, sep) for clean outputs
- Fallback mechanisms for invalid outputs:
  - Allows 1.5x max tries to get valid question-answer pairs
  - Duplicates last distractor if fewer than 3 are generated

## Model Configuration

- Question Generation: potsawee/t5-large-generation-race-QuestionAnswer
- Distractor Generation: potsawee/t5-large-generation-race-Distractor
- Answer Selection: potsawee/longformer-large-4096-answering-race

## Usage

```python
from selfcheck_metrics import SelfCheckMQAG

# Initialize with appropriate model paths
mqag = SelfCheckMQAG(...)

# Get inconsistency scores for sentences
scores = mqag.predict(sentences=sentences, samples=samples)
```

For more details, see the [original paper](https://arxiv.org/abs/2301.12307) and [implementation](https://github.com/potsawee/selfcheckgpt).
