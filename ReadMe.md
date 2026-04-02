# AI-Powered Invoice & Payment Delay Prediction System

## Overview
This project predicts whether an invoice is likely to be paid late using machine learning.

## Tech Stack
- Python
- Flask
- Pandas
- Scikit-learn

## Features
- Train ML model on invoice data
- Predict payment delay using REST API
- Return probability and risk category

## Dataset Columns
- invoice_amount
- payment_terms
- customer_score
- previous_delay_avg
- invoice_month
- days_to_due
- late_payment

## How to Run

### 1. Create virtual environment
```bash
python -m venv venv