# Adversarial Memory Adaptation Pipeline

A pipeline that autonomously constructs QA pairs and evaluates memory quality to support memory completion and memory construction strategy updating.

## Overview

This pipeline tests whether an AI memory system properly retains information from dialogue sessions. It generates questions from original dialogues, compares memory-based answers against ground truth, and provides improvement strategies when memory quality is insufficient.

## Project Structure

├── src/

│ ├── challenger.py # QA pair generation

│ ├── evaluator.py # Memory answering and quality evaluation

│ ├── adapter.py # Update strategy generation

│ └── AMA_pipeline.py # Main pipeline orchestration

├── data/

│ └── locomo10.json # Sample dialogue data

└── README.md

