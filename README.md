## **Impact of Quantization on a Question-Answering model**

This project measures how much accuracy a language model loses when its weights are compressed (quantized) to fewer bits. We evaluate Google's **Flan-T5** on the **AdversarialQA** question-answering dataset at 4 precision levels: **FP32** (full precision), **BF16**, **INT8** and **INT4**.

We study 2 questions:
- **Model size:** does quantization affect a small model (Flan-T5-base, 250M parameters) and a larger one (Flan-T5-large, 780M) in the same way?
- **Fine-tuning:** does a model fine-tuned on the task hold up better against quantization than the same model without fine-tuning?


## **[Live demo](https://flant5.sofianechaoui.com/)**

The project is deployed in production and can be tested online, with no installation: **[try the demo](https://flant5.sofianechaoui.com/)**.

The webapp has three tabs:

* **Live demo**: ask a question about a text and compare the raw and fine-tuned flan-t5-base side by side in FP32, BF16 and INT8, with the answer, latency and model size of each configuration.
* **Benchmark results**: F1 and Exact Match of the 12 evaluated configurations.
* **Stored predictions**: model answers on the test split, which can be loaded into the live demo.

It is self-hosted on a CPU server (Docker, HTTPS). See the [Deployment](#deployment) section for details.


## **About the dataset**

This project uses [AdversarialQA](https://huggingface.co/datasets/UCLNLP/adversarial_qa), an English extractive question-answering benchmark in the SQuAD 1.1 format. Each example pairs a Wikipedia passage with a question whose answer is a span of that passage.

It comes with a challenge due to how it was was built: annotators wrote questions with a reading-comprehension model in the loop (BiDAF, BERT-Large or RoBERTa-Large) and kept rewriting them until the model answered incorrectly. The result is a set of questions that strong QA models find hard, often requiring more reasoning than simple keyword matching.


## **Project Structure**

```bash
DeepLearning
│
├── app/
│   └── data/
│   └── favicon/
│   └── static/
│
├── data/
├── models/
│
├── notebooks/
│
├── results/
│
├── scripts/
│
├── src/
│
├── Dockerfile.txt
├── docker-compose.yaml
├── README.md
└── requirements.txt

```
* **app/**: web application for serving the model and interacting with it through a browser.
* **data/**: Empty placeholder. The dataset is downloaded directly from its source by the code in `notebooks/`.
* **models/**: Empty placeholder. Pretrained models & fine-tuned model are downloaded directly from their source by the code in `notebooks/`.
* **notebooks/**: Step-by-step Jupyter notebooks covering the full project pipeline.
* **results/**: Outputs produced such as saved model predictions.
* **scripts/**: Utility scripts that prepare data for the app.
* **src/**: Python modules shared across the notebooks.
* **Dockerfile.txt**: Instructions for building the Docker image that packages the app and its dependencies.
* **docker-compose.yaml**: Configuration for building and running the app container.
* **README.md**: Overview of the project, setup and usage.
* **requirements.txt**: Dependencies needed to run the project.


## **Reproducibility**

* The full methodology, results and discussion are described in a detailed report, which is not included in this repository. However, the whole project can be reproduced with the elements provided in this repository. 
* The dataset and models are downloaded automatically from the Hugging Face Hub. 
* A CUDA GPU is required for the INT8 and INT4 runs.
* The notebooks found in the `notebooks/` directory can be run using Google Colab (T4 recommended).
* To run locally, all packages are listed in `requirements.txt`.


## **Installation of the webapp**

The webapp runs on CPU only. On first launch, the models are downloaded from the Hugging Face Hub (about 2 GB), so the first start might take a few minutes.

### **With Docker**

Requirements: Docker and Docker Compose, about 8 GB of RAM.

```bash
git clone https://github.com/Julianlls/DeepLearning.git
cd DeepLearning
docker compose up -d --build
```

Then open http://localhost:7860

* Logs: `docker compose logs -f`
* Stop: `docker compose down`

### **Without Docker**

Requirements: Python 3.12. From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install gradio==6.28.0 -r app/requirements.txt
python -m app.app
```

Then open http://127.0.0.1:7860

Without Docker, models are loaded on the first question instead of at startup.


## **Deployment**

An interactive Gradio demo is self-hosted in Docker on a Proxmox server, behind a Caddy reverse proxy that handles HTTPS. It lets you ask questions to the raw and fine-tuned flan-t5-base in FP32, BF16 and INT8, and shows the answer, latency and model size for each precision. It also displays the benchmark results and the stored predictions.

* `app/`: the Gradio app, which reuses `src/config.py` (prompt, lengths) and `src/evaluate.py` (metrics)
* `Dockerfile`, `docker-compose.yml`: CPU-only production image. All models are preloaded at startup, and downloaded weights are kept in a volume.
* `.github/workflows/ci.yml`: on every push, smoke-tests the app and builds the image

**Why INT8 differs from the benchmark and INT4 is not live**

The benchmark quantizes with bitsandbytes (`load_in_8bit` / `load_in_4bit`), which needs an NVIDIA GPU with CUDA. The production server has no GPU, so:

* **INT8** uses PyTorch dynamic quantization instead. It is built into PyTorch, runs well on CPU, and behaves like the benchmark: answers are almost unchanged, and the model is about 3× smaller and faster. The app labels it as a CPU method.
* **INT4** is shown only through the measured benchmark results (charts and table), not live. CPU 4-bit methods exist (e.g. optimum-quanto, ONNX Runtime), but they are a different quantization scheme from bitsandbytes, so the live answers could disagree with the reported scores. On this CPU they would also likely be slower than FP32, because weights are unpacked at every step. Showing them would misrepresent the project's findings.

**Server limitations**

The demo runs on a self-hosted Proxmox server (Debian LXC container with Docker):

* **CPU only**: Intel Core i7-6700K (4 cores / 8 threads, 2015, AVX2 but no AVX-512/VNNI) and no GPU. This rules out bitsandbytes and fast low-bit kernels. flan-t5-large (780M parameters) is not served live either: on this CPU it would roughly triple the memory use and the response time.
* **Memory**: all six live configurations (2 models × FP32/BF16/INT8) are preloaded, which uses about 5 GB of RAM. The container is capped at 8 GB.
* **One request at a time**: generation is CPU-bound, so Gradio processes requests in a queue. Simultaneous visitors wait their turn.
* **Single machine on a home connection**: there is no redundancy, so availability depends on the server, the power supply and the home internet line.
* **Latency is indicative**: the timings in the app are measured on this CPU. On a GPU they would be much lower, and the gaps between precisions would be different.


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
