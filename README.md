
## **Impact of Quantization on a Question-Answering model**

**This project measures what that cost in accuracy. We evaluate an encoder-decoder model: flan-t5, on a generative question answering task at four precision levels and we look at two things: whether the degradation depends on model size, i.e. if a larger model would have a less important degradation of performance while precision decreases, and whether the degradation depends on fine-tuning on the task.**


## **About the dataset**

## **Project Structure**

```bash
diabetes_prediction
├── data/
├── models/
│
├── notebooks/
│   └── 01_EDA.ipynb
│   └── 02_pre_processing.ipynb
│   └── 03_baseline_evaluation.ipynb
│   └── 04_finetuning.ipynb
│   └── 05_finetuning_eval.ipynb
│
├── results/
│
├── src/
│
├── README.md
└── requirements.txt

```
* **data/**:
* **models/**:
* **notebooks/**: 
* **results/**:
* **src/**:
* **README.md**: provides an overview of the project.
* **requirements.txt**: lists all the dependencies required to run the project.


## **Reproducibility of the analysis**

The notebooks found in the `notebooks/` directory can be run from the top using Google Colab.
Predictions and metrics are written to `results/`.

To run locally, all packages are listed in `requirements.txt`.
Note that INT8 and INT4 runs require CUDA and will not work on Apple Silicon.

## **Installation of the webapp**
-


## **Contributors**
* [Julianlls](https://github.com/Julianlls)
* [MarcoDainese](https://github.com/MarcoDainese)
* [ernest-cyganowski](https://github.com/ernest-cyganowski)
* [Lucky12348](https://github.com/Lucky12348)



## **References**
* https://huggingface.co/google/flan-t5-base
* https://huggingface.co/google/flan-t5-large
* https://huggingface.co/datasets/UCLNLP/adversarial_qa
* https://huggingface.co/docs/bitsandbytes/en/index


--------------------------------------------------


## **Effect of quantization (FP32 -> INT4) with flan-t5 on AdversarialQA**

#### **Context**

Serving models on companies' infrastructure cost memory. As such, quantization allows to cuts the memory footprint at the cost of losing some of the precision and the range values that the parameters of the transformer can take. 


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

We fine-tune **flan-t5-base only**, in **FP32** on a Colab T4 (full fine-tuning, all parameters; gradient checkpointing to fit the T4, so no layer freezing was needed). We reuse the `load_processed()` splits (train 27k / val 3k) — **no re-split, no re-tokenization** — training directly on the dataset's `input_ids` / `attention_mask` / `labels` with the same prompt template as the baseline. A **single** training run is performed; quantization is applied afterwards to this checkpoint. The learning rate is selected on **val F1** (computed with `evaluate.py`, greedy, `max_new_tokens=48`) over {5e-5, 1e-4, 3e-4} → **5e-5, 3 epochs**. The fine-tuned checkpoint is published on the Hugging Face Hub: `Smambu/flan-t5-base-adversarialqa-ft`.

#### **Results**


| Model | Precision | EM | token_precision | token_recall | F1 | F1 vs.FP32 |
|---|---|---|---|---|---|---|
| flan-t5-base | FP32 | 0.4237 | 0.5587 | 0.5553 | 0.5333 | - |
| flan-t5-base | BF16 | 0.4257 | 0.5600 | 0.5562 | 0.5342 | +0.001 |
| flan-t5-base | INT8 | 0.4220 | 0.5571 | 0.5554 | 0.5323 | -0.001 |
| flan-t5-base | INT4 | 0.4033 | 0.5404 | 0.5249 | 0.5094 | −0.024 |
| flan-t5-large | FP32 | 0.5483 | 0.6842 | 0.6968 | 0.6639 | - |
| flan-t5-large | BF16 | 0.5490 | 0.6855 | 0.6983 | 0.6650 | +0.001 |
| flan-t5-large | INT8 | 0.5490 | 0.6848 | 0.6984 | 0.6647 | +0.001 |
| flan-t5-large | INT4 | 0.5303 | 0.6687 | 0.6806 | 0.6476 | −0.016 |
| flan-t5-base-finetuned | FP32 | 0.4440 | 0.5674 | 0.5840 | 0.5501 | - |
| flan-t5-base-finetuned | BF16 | 0.4423 | 0.5661 | 0.5838 | 0.5490 | −0.001 |
| flan-t5-base-finetuned | INT8 | 0.4453 | 0.5674 | 0.5856 | 0.5513 | +0.001 |
| flan-t5-base-finetuned | INT4 | 0.4340 | 0.5606 | 0.5618 | 0.5379 | −0.012 |

#### **Interpretation prior fine-tuning**

* **BF16 and INT8 are free** for both models. BF16 and INT8 have +/- 0.001 F1 of the baseline FP32. We can see that BF16 has an even higher F1, on both models. Differences of such size likely reflect numerical noise, not a degredation due to reduced precision. Halving the weights with BF16 or quartering it with INT8 leaves answer quality unchanged on this task. 

* **INT4 is the first cost**. It is the only precision level where there is a drop, in both models: -0.024 F1 for the base (-4.5%) and -0.016 F1 for the large (-2.5%). The loss is measurable but still modest, meaning that INT4 keeps roughly 95% of the baseline quality at a quarter of the storage. 


#### **Interpretation post fine-tuning**

* **Fine-tuning helps, but modestly.** Measured on the **test** split, F1 goes from 0.533 (raw base) to **0.550** (fine-tuned) and EM from 0.424 to **0.444** (+0.017 F1, +0.020 EM). The gain is small because flan-t5-base is already instruction-tuned for extractive QA and AdversarialQA is adversarial by design, so the headroom is limited; the model does not degrade.

* **The gain must be read on test, not on val.** The raw base scores 0.718 F1 on our `val` but only 0.533 on `test`. Our `val` is carved from the original train (which flan-t5's instruction tuning most likely already covered), so it is contaminated and saturated — it overstates performance and cannot show the fine-tuning effect. The held-out `test` (the original validation split) is the honest yardstick.

* **Quantization of the fine-tuned model** (BF16 / INT8 / INT4): *to be completed with the quantization grid.* The key comparison is whether the per-precision F1 drop matches the one measured on the raw base (e.g. INT4 −0.024 F1).

#### **Repository structure**


notebooks/   01_eda, 02_preprocessing, 03_baseline_eval, 04_finetuning, 05_finetuning_eval
src/         config.py preprocessing.py evaluate.py
data/        generated by the pipeline, not tracked

`src/` holds the functions reused across notebooks, so every grid cell is scored by the same code. Notebooks 02 and 03 also show how those functions were built, step by step, before being extracted.


#### **Deployment (live demo)**

An interactive Gradio demo is self-hosted in Docker on a Proxmox server, behind a Caddy reverse proxy that handles HTTPS. It lets you ask questions to the raw and fine-tuned flan-t5-base in FP32, BF16 and INT8, and shows the answer, latency and model size for each precision. It also displays the benchmark results and the stored predictions.

* `app/`: the Gradio app, which reuses `src/config.py` (prompt, lengths) and `src/evaluate.py` (metrics)
* `Dockerfile`, `docker-compose.yml`: CPU-only production image. All models are preloaded at startup, and downloaded weights are kept in a volume.
* `.github/workflows/ci.yml`: on every push, smoke-tests the app and builds the image

bitsandbytes needs CUDA, so the live INT8 option uses PyTorch dynamic quantization on CPU. The INT4 figures come from the benchmark run.

Run with Docker: `docker compose up -d --build`, then open http://localhost:7860.
Run without Docker, from the repo root: `pip install gradio -r app/requirements.txt`, then `python -m app.app`.

#### **Limitations**

* zero-shot scores on an adversarial dataset, made difficult to solve on purpose.
* the dataset was initially made for extractive QA. Even though flan-t5 was fined-tuned with extractive QA datasets of type SQuAD
* no memory footprint and inference latency were measured, so the accuracy/cost tradeoff is only partially measured
* generative QA on an extractive task. The consequence is that a semantically correct answer generated differently will score as wrong
