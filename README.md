# RL_HW2 — Evolution Strategies (ES) Experiments

This repository contains my implementation and analysis for **HW2 (Programming)**, using **Evolution Strategies (ES)** to optimize a policy and study how hyperparameters affect learning curves.

It includes:
- ES training code (`es_trainer.py`)
- Runners to reproduce the **5-seed** category plots and the **15-run** best-config plot
- CSV logs for analysis and the final figures used in the write-up

> **TL;DR**: run `runner_final.py` to reproduce results; logs are written to CSV and plots can be generated from those logs.

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Requirements](#requirements)
- [Project Layout](#project-layout)
- [What Each File Does](#what-each-file-does)
- [How to Run](#how-to-run)
  - [A) Reproduce 5-seed learning curves for selected combos](#a-reproduce-5-seed-learning-curves-for-selected-combos)
  - [B) Reproduce the 15-run “best config” curve (Q2.2)](#b-reproduce-the-15-run-best-config-curve-q22)
- [Plotting Tips](#plotting-tips)
- [Repro Notes](#repro-notes)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Citation / Attribution](#citation--attribution)

---

## Environment Setup

Use a clean virtual environment (conda or venv).

```bash
# Using conda
conda create -n rl-hw2 python=3.10 -y
conda activate rl-hw2

# Or with venv
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

