## **Effect of quantization (FP32 -> INT4) with flan-t5 on AdversarialQA**

#### **Context**

Serving models on companies' infrastructure cost memory. As such, quantization allows to cuts the memory footprint at the cost of losing some of the precision and the range values that the parameters of the transformer can take. 

This project measures what that cost in accuracy. We evaluate an encoder-decoder model: flan-t5, on a generative question answering task at four precision levels and we look at two things: whether the degradation depends on model size, i.e. if a larger model would have a less important degradation of performance while precision decreases, and whether the degradation depends on fine-tuning on the task.

#### **Experimental setup**

**Models**: We use the reference encoder-decoder from Google, coming from the initial T5, but that had already been fine-tuned on some tasks in addition to their initial pre-training: Flan-T5-base (250M parameters) and Flan-T5-large (780M). We chose these models because they are already instruction-tuned, meaning that the raw models can generate something usable in zero-shot (without requiring initial fine-tuning just to use them for a QA task). Both models have the same tokenizer and the same vocabulary, meaning that we can reuse the tokenized data for both, allowing easier comparison. The flan-t5-large is used to compare the model performance on different precision level (FP32, BF16, INT8, INT4) with the flan-t5-base only prior fine tuning. As it is a more consequent model, flan-t5-base was the only model fine tuned. Therefore, the large model's use is mainly to have a quick look at the impact on the quality of the model's prediction at different precision level, compared to the same "lighter" model on equivalent precision level, prior fine tuning.

Then, after fine-tuning flan-t5-base, we will re-assess the different precision levels impact on the base fine-tuned base model only. 

**Dataset**: for this project we'll use the the AdversarialQA dataset, as it is a challenging dataset since the exampels were written by humans that were trying to fool a model. One thing that is important to know, is that this dataset was made initially for extractive QA with encoder-only models. 

The splits are the following: 27k/3k/3k train/val/test, done with 42 as the split seed for reproducibility.

**Precisions levels** tested are FP32, BF16, INT8, INT4 (INT8 and INT4 through bitsandbytes). FP16 was excluded because T5 was pre-trained in BF16. FP16 has a much narrower range, and because of that some T5 activations functions fall outside of it and it would result in a failure of the model.

**States**: 
* `zero-shot` where we compare base and large on different precision levels 
* `fine-tuned` where we compare the different performance of the base model only on different precision levels. 

**Assessed**: 2 × 4 (zero-shot) + 4 (fine-tuned) = 12 configurations

**Metrics**: Exact Match and token-level F1, SQuAD-style normalisation, scores on a 0–1 scale

**Hardware**: Colab T4 for the quantized runs (bitsandbytes needs CUDA) is enough for the base model. Though, for the flan-t5-large a stronger hardware config is more suitable and we ran the comparison on colab A100 hardware setup. T4 would work, but it would take longer.

**Generation config**: 
* `max_input_length = 512`, as it is a legacy from prior training of the t5 family, we left it as it since.following the EDA, the count of inputs (question + context) having a length > 512 tokens were marginal.
* `max_new_tokens = 48`. We prefered keeping the generating output limited by the model as it is a generative model assessed on an extractive QA dataset.
* `greedy decoding` was used for generation, no sampling.


#### **Fine-tuning**

XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

#### **Results**


| Model | Precision | EM | token_precision | token_recall | F1 | <> F1 vs.FP32 |
|---|---|---|---|---|---|---|
| flan-t5-base | FP32 | 0.4237 | 0.5587 | 0.5553 | 0.5333 | - |
| flan-t5-base | BF16 | 0.4257 | 0.5600 | 0.5562 | 0.5342 | +0.001 |
| flan-t5-base | INT8 | 0.4220 | 0.5571 | 0.5554 | 0.5323 | -0.001 |
| flan-t5-base | INT4 | 0.4033 | 0.5404 | 0.5249 | 0.5094 | −0.024 |
| flan-t5-large | FP32 | 0.5483 | 0.6842 | 0.6968 | 0.6639 | - |
| flan-t5-large | BF16 | 0.5490 | 0.6855 | 0.6983 | 0.6650 | +0.001 |
| flan-t5-large | INT8 | 0.5490 | 0.6848 | 0.6984 | 0.6647 | +0.001 |
| flan-t5-large | INT4 | 0.5303 | 0.6687 | 0.6806 | 0.6476 | −0.016 |
| flan-t5-base-finetuned | FP32 | XXXX | XXXX | XXXX | XXXX | XXXX |
| flan-t5-base-finetuned | BF16 | XXXX | XXXX | XXXX | XXXX | XXXX |
| flan-t5-base-finetuned | INT8 | XXXX | XXXX | XXXX | XXXX | XXXX |
| flan-t5-base-finetuned | INT4 | XXXX | XXXX | XXXX | XXXX | XXXX |

#### **Interpretation prior fine-tuning**

* **BF16 and INT8 are free** for both models. BF16 and INT8 have +/- 0.001 F1 of the baseline FP32. We can see that BF16 has an even higher F1, on both models. Differences of such size likely reflect numerical noise, not a degredation due to reduced precision. Halving the weights with BF16 or quartering it with INT8 leaves answer quality unchanged on this task. 

* **INT4 is the first cost**. It is the only precision level where there is a drop, in both models: -0.024 F1 for the base (-4.5%) and -0.016 F1 for the large (-2.5%). The loss is measurable but still modest, meaning that INT4 keeps roughly 95% of the baseline quality at a quarter of the storage. 


#### **Interpretation post fine-tuning**

XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

#### **Repository structure**


notebooks/   01_eda, 02_preprocessing, 03_baseline_eval, 04_finetuning, 05_finetuning_eval
src/         config.py preprocessing.py evaluate.py
data/        generated by the pipeline, not tracked

`src/` holds the functions reused across notebooks, so every grid cell is scored by the same code. Notebooks 02 and 03 also show how those functions were built, step by step, before being extracted.

#### **How to reproduce**

On Colab, open `03_baseline_eval.ipynb` and run it from the top: the first cells clone the repository and regenerate the preprocessed dataset, so there is no need to run 01 and 02 first. Predictions and metrics are written to `results/`.
 
Expect a noticeable runtime on the baseline evaluation. 
 
To run locally, all packages are listed in `requirements.txt`. Note that INT8 and INT4 runs require CUDA and will not work on Apple Silicon.


#### **Limitations**

* zero-shot scores on an adversarial dataset, made difficult to solve on purpose.
* the dataset was initially made for extractive QA. Even though flan-t5 was fined-tuned with extractive QA datasets of type SQuAD
* no memory footprint and inference latency were measured, so the accuracy/cost tradeoff is only partially measured
* generative QA on an extractive task. The consequence is that a semantically correct answer generated differently will score as wrong


#### **References**

* https://huggingface.co/google/flan-t5-base
* https://huggingface.co/google/flan-t5-large
* https://huggingface.co/datasets/UCLNLP/adversarial_qa
* https://huggingface.co/docs/bitsandbytes/en/index
