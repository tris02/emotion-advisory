# Installation Guide

## Prerequisites
Ensure you have the following dependencies installed before proceeding:

- **Python 3.8+**
- **pip** (Python package manager)
- **Git** (to clone the repository)

## Steps

### 1️. Clone the Repository
```bash
git clone https://github.com/tris02/emotion-advisory.git
cd emotion-advisory
```

### 2️. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate  # For Windows
```

### 3️. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️. Install the Package
```bash
pip install .
```

## Verifying Installation
To ensure the package is installed correctly, run:
```bash
python -m emotion_advisory.main
```

## Troubleshooting
- If `pip` fails, ensure it is upgraded:
```bash
pip install --upgrade pip
```
- If dependencies conflict, try using a virtual environment.

## Uninstallation
To remove the package, run:
```bash
pip uninstall emotion-advisory
```

