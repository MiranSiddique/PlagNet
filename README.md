# PlagNet

## Overview

This tool provides a graphical user interface (GUI) built with Tkinter to help users compare text files for similarity. It's particularly useful for identifying potential plagiarism or finding near-duplicate documents. The tool supports multiple sophisticated similarity calculation methods, including a knowledge-graph-based approach using NLTK and WordNet, and various state-of-the-art Transformer models (like BERT, RoBERTa, BART, T5, Sentence-Transformers) via the Hugging Face `transformers` library.

Users can register and log in to manage their own sets of files securely. The tool calculates pairwise similarity between uploaded files based on the selected method and threshold, displaying pairs that exceed the specified similarity percentage.

## Features

*   **User Authentication:** Secure user registration and login system. User files are stored separately.
*   **File Management:** Upload text-based files (`.txt`, `.py`, `.md`, etc.) associated with the logged-in user.
*   **Multiple Similarity Methods:** Choose from:
    *   **WordNet (Improved):** A semantic similarity approach using NLTK's WordNet, considering synonyms, hypernyms/hyponyms, and performing bidirectional comparison for robustness.
    *   **Sentence Transformers:** Models optimized for generating sentence embeddings (e.g., `all-MiniLM-L6-v2`, `all-mpnet-base-v2`).
    *   **General Transformers:** Standard Transformer models like BERT, RoBERTa, DistilBERT.
    *   **Seq2Seq Models (Encoder):** Using the encoder part of models like BART and T5 for embeddings.
*   **Customizable Threshold:** Set the similarity percentage (0-100) above which file pairs should be reported.
*   **Results Display:** Clearly lists pairs of similar files and their calculated similarity score.
*   **Model Caching:** Loads computationally expensive Transformer models only once per session for efficiency.
*   **GPU Acceleration:** Automatically utilizes CUDA-enabled GPU if detected via PyTorch, significantly speeding up Transformer-based comparisons.
*   **Cross-Platform GUI:** Built with Python's standard Tkinter library.

## Setup and Installation

Follow these steps to set up and run the project:

**1. Prerequisites:**

*   **Python:** Python 3.8 or higher is recommended.
*   **pip:** Python package installer (usually comes with Python).
*   **Git:** (Optional) For cloning the repository.
*   **Tkinter:** Usually included with Python on Windows/macOS. On Linux, you might need to install it separately:
    *   Debian/Ubuntu: `sudo apt-get update && sudo apt-get install python3-tk`
    *   Fedora: `sudo dnf install python3-tkinter`

**2. Clone or Download the Repository:**

```bash
git clone https://github.com/MiranSiddique/PlagNet.git
```

**3. Install The Dependencies mentioned in requirements.txt:**
```bash
pip install -r requirements.txt
```
**4. Run the Application:**
```bash
python main.py
```
