# How to Run the Project

## 1. Create Virtual Environment

```bash
python -m venv venv
```

## 2. Activate

### Windows

```bash
venv\Scripts\activate
```

### Mac/Linux

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Train Models

```bash
python train.py
```

## Outputs Generated

* outputs/results.csv
* outputs/tradeoff.png
* outputs/gate_histogram.png
* checkpoints/*.pth
