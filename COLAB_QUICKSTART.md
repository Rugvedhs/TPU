# Colab Quickstart

Use a Google Colab notebook with a `T4 GPU`.

## 1. Clone the repo

```python
!git clone https://github.com/Rugvedhs/TPU.git
%cd /content/TPU
```

## 2. Check the runtime

```python
!nvidia-smi
!python --version
```

## 3. Install dependencies

```python
!pip install -r requirements.txt
```

## 4. Generate traces

```python
!python scripts/run_profile.py --num-runs 24
```

## 5. Train the policy

```python
!python scripts/run_train.py
```

## 6. Evaluate baselines vs learned policy

```python
!python scripts/run_eval.py
```

## 7. Inspect outputs

```python
!find results -maxdepth 3 -type f | sort
!find data -maxdepth 3 -type f | sort
```

## What To Send Back If Something Breaks

Send me:

- the full error text
- the output of `!nvidia-smi`
- the output of `!python --version`
- which cell failed
