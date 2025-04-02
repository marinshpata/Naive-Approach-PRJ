# Naive-Approach-PRJ
This folder contains all the code and instructions necessary to run my Naive Approach experiment as part of my 6CCS3PRJ Final Year Project.

## Data Instructions

Download all datasets required for domain adaptation from:

https://drive.google.com/drive/folders/1FohHF7i3mX8mlIwYU-7VhrstQKn7sVuq?usp=share_link

This was necessary because of the submission file limitations and also limitations on github.
Once downloaded, place all datasets directly in a **dataset** folder in the **root directory**.  
This is essential for the `fineTune.py` script.

## Setup Instructions

### 1. Python Version

Ensure **Python 3.8 or newer** is installed.

### 2. Create a Virtual Environment (Recommended)

```bash
# Create the environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Experiment
Make sure you are inside the repository root (`Naive-Approach-PRJ` or the renamed version):

```bash
cd "Naive-Approach-PRJ"
```
or
```bash
cd "Naive Approach"
```
Then run the fine-tuning script with:

```bash
python fineTune.py
```

**Important:**

- The experiment can take several hours to run.
- Connect your computer to power.

## Experiment Outputs

After training, the models and results are saved to the following directories:

```bash
./domain_adapt_mlm/dictionary/ 
./glue_sst2_dictionary/ 
./glue_baseline_sst2/
etc...
```
Metrics such as accuracy and loss will be printed in the terminal.

## ⚠️ Disclaimer

**Note:** Results obtained from running this code may differ from the results reported in the final project report due to hardware differences and the inherent nature of some non deterministic GPU operations.

## Complete Dependency List

Expand below to view all required packages with exact versions:

<details>
  <summary>Full list of required Python packages</summary>

  ```text
  accelerate==1.2.1
  aiohappyeyeballs==2.4.4
  aiohttp==3.11.11
  aiosignal==1.3.2
  attrs==24.3.0
  certifi==2024.12.14
  charset-normalizer==3.4.1
  datasets==3.2.0
  dill==0.3.8
  evaluate==0.4.3
  filelock==3.16.1
  frozenlist==1.5.0
  fsspec==2024.9.0
  huggingface-hub==0.27.1
  idna==3.10
  inquirerpy==0.3.4
  Jinja2==3.1.5
  joblib==1.4.2
  MarkupSafe==3.0.2
  mpmath==1.3.0
  multidict==6.1.0
  multiprocess==0.70.16
  networkx==3.4.2
  numpy==2.2.1
  packaging==24.2
  pandas==2.2.3
  pfzy==0.3.4
  pip==24.0
  prompt_toolkit==3.0.50
  propcache==0.2.1
  psutil==6.1.1
  pyarrow==18.1.0
  python-dateutil==2.9.0.post0
  pytz==2024.2
  PyYAML==6.0.2
  regex==2024.11.6
  requests==2.32.3
  safetensors==0.5.1
  scikit-learn==1.6.1
  scipy==1.15.2
  setuptools==65.5.0
  six==1.17.0
  sympy==1.13.1
  threadpoolctl==3.5.0
  tokenizers==0.21.0
  torch==2.5.1
  tqdm==4.67.1
  transformers==4.47.1
  typing_extensions==4.12.2
  tzdata==2024.2
  urllib3==2.3.0
  wcwidth==0.2.13
  xxhash==3.5.0
  yarl==1.18.3
```
</details>

