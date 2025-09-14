import os
import hashlib
import json
import nltk
import numpy as np

from nltk.corpus import stopwords, wordnet as wn
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import time # For timing model loading
import torch
import torch.nn.functional as NNF
from sklearn.metrics.pairwise import cosine_similarity

# Using try-except for optional dependencies
try:
    from transformers import (
        AutoTokenizer, AutoModel,
        BartTokenizer, BartModel,
        T5Tokenizer, T5Model
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: 'transformers' library not found. Transformer-based similarity methods will be unavailable.")
    print("Install it using: pip install transformers torch")

try:
    pass # Already handled by AutoModel
except ImportError:
    # This check might be redundant if transformers is installed,
    # but good practice if someone installs only parts.
    print("Warning: 'sentence-transformers' might be needed for some models if AutoModel fails.")
    print("Install it using: pip install sentence-transformers")

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# --- Defining Model Choices ---
MODEL_CHOICES = {
    "WordNet (Graph-based)": "wordnet",
    "Sentence Transformers": [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        # "sentence-transformers/paraphrase-MiniLM-L6-v2", # Similar to all-MiniLM
        "sentence-transformers/all-distilroberta-v1",
        # "sentence-transformers/stsb-roberta-large", # Larger model
        # "sentence-transformers/multi-qa-mpnet-base-dot-v1", # Tuned for QA
        # "intfloat/e5-large", # Larger model
        # "BAAI/bge-large-en" # Larger model
    ],
    "General Transformers": [
        "bert-base-uncased",
        # "bert-large-uncased", # Larger model
        "roberta-base",
        # "roberta-large", # Larger model
        "distilbert-base-uncased",
    ],
    "Seq2Seq Models (Encoder)": [
        "google/flan-t5-base",
        # "google/flan-t5-large", # Larger model
        "facebook/bart-base",
        # "facebook/bart-large", # Larger model
    ]
}

# Flatten the choices for the dropdown
ALL_MODEL_NAMES = ["WordNet (Graph-based)"]
if TRANSFORMERS_AVAILABLE:
    for category, models in MODEL_CHOICES.items():
        if category != "WordNet (Graph-based)":
            if isinstance(models, list):
                ALL_MODEL_NAMES.extend(models)
            else: # Should not happen with current structure, but for safety
                ALL_MODEL_NAMES.append(models)
else:
    print("\nTransformer models disabled due to missing libraries.\n")


class FileSimilarityTool:
    def __init__(self):
        self.users = {}
        self.user_files = {}
        self.current_user = None
        self.data_dir = Path("data")
        self.users_file = self.data_dir / "users.json"

        # --- New Attributes for Model Caching ---
        self.loaded_model_name = None
        self.loaded_tokenizer = None
        self.loaded_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        # --- End New Attributes ---

        self.init_directories()
        self.load_users()
        self.setup_ui()

    def init_directories(self):
        """Initialize necessary directories for data storage"""
        self.data_dir.mkdir(exist_ok=True)
        (self.data_dir / "files").mkdir(exist_ok=True)

        if not self.users_file.exists():
            with open(self.users_file, 'w') as f:
                json.dump({}, f)

    def load_users(self):
        """Load user data from the JSON file"""
        try:
            with open(self.users_file, 'r') as f:
                self.users = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            self.users = {}

    def hash_password(self, password):
        """Hash a password for storage"""
        return hashlib.sha256(password.encode()).hexdigest()

    def register_user(self, username, password):
        """Register a new user"""
        if username in self.users:
            return False, "Username already exists"

        self.users[username] = {
            "password_hash": self.hash_password(password),
            "files": []
        }
        self.save_users()

        if username not in self.user_files:
             self.user_files[username] = []
        return True, "User registered successfully"

    def login_user(self, username, password):
        """Login an existing user"""
        if username not in self.users:
            return False, "User does not exist"

        if self.users[username]["password_hash"] != self.hash_password(password):
            return False, "Incorrect password"

        self.current_user = username
        # Loads files for the current user into self.user_files if not already there
        # ensuring consistency after login
        if self.current_user not in self.user_files:
            self.user_files[self.current_user] = self.users[self.current_user].get("files", [])
        return True, "Login successful"

    def save_users(self):
        """Save user data to the JSON file"""
        # Ensure file lists are consistent before saving
        if self.current_user and self.current_user in self.users:
             self.users[self.current_user]['files'] = self.get_user_files() # Get from user_files cache

        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=4) 

    def user_directory(self, username):
        """Get the directory for a specific user's files"""
        user_dir = self.data_dir / "files" / username
        user_dir.mkdir(exist_ok=True)
        return user_dir

    def upload_file(self, file_path):
        """Upload a file for the current user"""
        if not self.current_user:
            return False, "No user logged in"

        try:
            file_name = Path(file_path).name
            user_dir = self.user_directory(self.current_user)
            dest_path = user_dir / file_name

            # Basic check for text-like files (can be improved)
            if not file_path.lower().endswith(('.txt', '.py', '.md', '.json', '.csv', '.html', '.xml')):
                 print(f"Warning: Uploading non-standard text file: {file_name}. Ensure it's UTF-8 encoded.")

            # Reading the file content
            try:
                with open(file_path, 'r', encoding='utf-8') as src_file:
                    content = src_file.read()
            except UnicodeDecodeError:
                 return False, f"Error reading {file_name}: File is not valid UTF-8 text."
            except Exception as e:
                 return False, f"Error reading {file_name}: {str(e)}"


            # Saving the file in the user's directory
            with open(dest_path, 'w', encoding='utf-8') as dest_file:
                dest_file.write(content)

            # Updating user's files list (using the self.user_files cache)
            if self.current_user not in self.user_files:
                self.user_files[self.current_user] = []
            if file_name not in self.user_files[self.current_user]:
                self.user_files[self.current_user].append(file_name)
                self.save_users() # Saves changes to the JSON file

            return True, f"File {file_name} uploaded successfully"
        except Exception as e:
            return False, f"Error uploading file: {str(e)}"

    def get_user_files(self):
        """Get the list of files for the current user from the cache"""
        if not self.current_user:
            return []
        # Ensure the user exists in the cache, load from self.users if not
        if self.current_user not in self.user_files:
             if self.current_user in self.users:
                 self.user_files[self.current_user] = self.users[self.current_user].get("files", [])
             else:
                 self.user_files[self.current_user] = [] # Should not happen if login logic is correct

        return self.user_files.get(self.current_user, [])


    def read_file_content(self, filename):
        """Read the content of a file"""
        if not self.current_user:
            self.status_label.config(text="Error: No user logged in.")
            return None

        file_path = self.user_directory(self.current_user) / filename
        if not file_path.exists():
            self.status_label.config(text=f"Error: File not found - {filename}")
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.status_label.config(text=f"Error reading {filename}: {str(e)}")
            return None

    # --- WordNet Similarity Logic ---
    def _calculate_wordnet_similarity(self, sentence1, sentence2):
        """Compare two sentences using the WordNet semantic graph approach."""
        try:
            # Creating semantic graphs for both sentences (Graph creation only, no visualization)
            _, words1, _ = self._create_semantic_graph_data(sentence1)
            _, words2, _ = self._create_semantic_graph_data(sentence2)

            # Calculating similarities...
            similarities = []
            for word1, synsets1 in words1.items():
                word_sims = []
                for word2, synsets2 in words2.items():
                    sim = self._calculate_word_similarity_wordnet(synsets1, synsets2)
                    word_sims.append(sim)
                if word_sims:
                    # Using max similarity found for word1 against all words in sentence2
                    similarities.append(max(word_sims))

            if not similarities or not words1 or not words2: # Handle empty sentences or no common words
                return 0.0

            # Simple averaging (can be refined later)
            # Calculate average similarity based on words in the *first* sentence
            avg_similarity = sum(similarities) / len(words1) if words1 else 0.0

            # Alternative: Symmetrical average
            # similarities_rev = []
            # for word2, synsets2 in words2.items():
            #     word_sims = []
            #     for word1, synsets1 in words1.items():
            #          sim = self._calculate_word_similarity_wordnet(synsets2, synsets1) # Reversed args
            #          word_sims.append(sim)
            #     if word_sims:
            #         similarities_rev.append(max(word_sims))
            # avg_similarity_rev = sum(similarities_rev) / len(words2) if words2 else 0.0
            # avg_similarity = (avg_similarity + avg_similarity_rev) / 2.0

            # Basic scaling heuristic (can be refined)
            coverage = len(similarities) / max(len(words1), len(words2)) if words1 and words2 else 0.0
            if coverage > 0.8 and avg_similarity > 0.7:
                final_score = min(1.0, avg_similarity * 1.2)
            else:
                final_score = avg_similarity

            return final_score * 100 # Scale to 0-100
        except Exception as e:
            print(f"Error during WordNet similarity calculation: {e}")
            return 0.0 # Return 0 on error

    def _create_semantic_graph_data(self, sentence):
        """Extracts word synsets from a sentence for WordNet comparison."""
        # No graph building needed, just extract synsets
        words = nltk.word_tokenize(sentence.lower())
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words] # Added isalnum
        tagged_words = nltk.pos_tag(filtered_words)

        word_synsets = {}
        for word, pos in tagged_words:
            # Map NLTK POS tags to WordNet POS tags
            wn_pos = None
            if pos.startswith('N'):
                wn_pos = wn.NOUN
            elif pos.startswith('V'):
                wn_pos = wn.VERB
            elif pos.startswith('J'):
                wn_pos = wn.ADJ
            elif pos.startswith('R'):
                 wn_pos = wn.ADV # Adverbs might be useful too

            if wn_pos:
                synsets = wn.synsets(word, pos=wn_pos)
                if synsets:
                    # Use a unique key like word_pos to avoid overwriting if same word appears with different POS
                    word_key = f"{word}_{wn_pos}"
                    if word_key not in word_synsets:
                         word_synsets[word_key] = []
                    word_synsets[word_key].extend(synsets) # Store all synsets for the word/pos

        # Return None for graph and types, just the synsets
        return None, word_synsets, None

    def _calculate_word_similarity_wordnet(self, synsets1, synsets2):
        """Calculate similarity between two sets of synsets using WordNet path similarity."""
        max_sim = 0.0
        if not synsets1 or not synsets2:
            return 0.0

        for syn1 in synsets1:
            for syn2 in synsets2:
                # Using path_similarity - robust measure
                sim = syn1.path_similarity(syn2)
                if sim is not None and sim > max_sim:
                    max_sim = sim

        # Alternative: wup_similarity (Wu-Palmer) ]
        # for syn1 in synsets1:
        #     for syn2 in synsets2:
        #         sim = syn1.wup_similarity(syn2)
        #         if sim is not None and sim > max_sim:
        #             max_sim = sim

        return max_sim
    # --- End WordNet Logic ---


    # --- Transformer Similarity Logic ---
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        """Mean Pooling - Take attention mask into account for correct averaging."""
        # model_output can be BaseModelOutputWithPoolingAndCrossAttentions or just tuple/tensor
        # We usually want the last_hidden_state which is typically the first element.
        if isinstance(model_output, tuple):
            token_embeddings = model_output[0]
        else:
            # Assuming it's an object with last_hidden_state (like BaseModelOutput)
            token_embeddings = model_output.last_hidden_state

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def load_similarity_model(self, model_name):
        """Loads the specified tokenizer and model, caching them."""
        if not TRANSFORMERS_AVAILABLE:
            messagebox.showerror("Error", "Transformers library not installed. Cannot use this model.")
            return False

        if model_name == self.loaded_model_name:
            # print(f"Model '{model_name}' already loaded.")
            return True # Already loaded

        self.status_label.config(text=f"Loading model: {model_name}... Please wait.")
        self.root.update_idletasks() # Update UI to show status

        start_time = time.time()
        try:
            # Clear previous model from memory (important for GPU)
            if self.loaded_model:
                del self.loaded_model
                self.loaded_model = None
            if self.loaded_tokenizer:
                 del self.loaded_tokenizer
                 self.loaded_tokenizer = None
            if 'cuda' in self.device:
                 torch.cuda.empty_cache()

            # --- Determine Tokenizer and Model Class ---
            # Default to AutoClass
            tokenizer_class = AutoTokenizer
            model_class = AutoModel

            # Special handling for known Seq2Seq models if needed
            if model_name.startswith("google/flan-t5"):
                tokenizer_class = T5Tokenizer
                model_class = T5Model # Use T5Model to get encoder access easily
            elif model_name.startswith("facebook/bart"):
                tokenizer_class = BartTokenizer
                model_class = BartModel

            print(f"Loading {model_name} using {tokenizer_class.__name__} and {model_class.__name__}...")

            self.loaded_tokenizer = tokenizer_class.from_pretrained(model_name)
            self.loaded_model = model_class.from_pretrained(model_name).to(self.device)
            self.loaded_model.eval() # Set model to evaluation mode
            self.loaded_model_name = model_name

            end_time = time.time()
            load_time = end_time - start_time
            self.status_label.config(text=f"Model '{model_name}' loaded in {load_time:.2f}s. Ready.")
            print(f"Model '{model_name}' loaded successfully to {self.device} in {load_time:.2f} seconds.")
            return True

        except Exception as e:
            self.status_label.config(text=f"Error loading model {model_name}: {e}")
            messagebox.showerror("Model Loading Error", f"Failed to load model '{model_name}'.\nError: {e}\n\nCheck model name and internet connection.")
            self.loaded_model = None
            self.loaded_tokenizer = None
            self.loaded_model_name = None
            return False

    def _calculate_transformer_similarity(self, sentence1, sentence2):
        """Calculates similarity using the currently loaded transformer model."""
        if not self.loaded_model or not self.loaded_tokenizer:
            self.status_label.config(text="Error: No model loaded.")
            # Attempt to load the last selected model? Or just error out.
            # For now, error out. User should click Compare again.
            # Maybe try loading based on self.method_var.get()?
            selected_method_name = self.method_var.get()
            if selected_method_name and selected_method_name != "WordNet (Graph-based)":
                 if not self.load_similarity_model(selected_method_name):
                     return 0.0 # Loading failed
                 # If loading succeeded, continue
            else:
                 messagebox.showerror("Error", "No similarity model is loaded. Please select one and click 'Compare Files'.")
                 return 0.0

        # print(f"Calculating similarity using: {self.loaded_model_name}")
        try:
            # Tokenize sentences
            encoded_input = self.loaded_tokenizer([sentence1, sentence2], padding=True, truncation=True, return_tensors='pt').to(self.device)

            # Compute token embeddings
            with torch.no_grad():
                # Handle different model architectures
                if isinstance(self.loaded_model, T5Model):
                    # T5/Flan-T5: Use encoder output directly
                     model_output = self.loaded_model.encoder(**encoded_input)
                elif isinstance(self.loaded_model, BartModel):
                     # BART: Get encoder's last hidden state
                     outputs = self.loaded_model(**encoded_input, output_hidden_states=True)
                     model_output = outputs.encoder_last_hidden_state # Access directly
                     # Need to structure it like the mean_pooling expects (a tuple or object with .last_hidden_state)
                     # Let's wrap it in a simple structure if it's just a tensor
                     if isinstance(model_output, torch.Tensor):
                         from types import SimpleNamespace
                         model_output = SimpleNamespace(last_hidden_state=model_output)

                else:
                    # General models (BERT, RoBERTa, DistilBERT, SentenceTransformers via AutoModel)
                    model_output = self.loaded_model(**encoded_input)

            # Perform pooling
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

            # Normalize embeddings
            sentence_embeddings = NNF.normalize(sentence_embeddings, p=2, dim=1)

            # Compute cosine similarity
            similarity = cosine_similarity(sentence_embeddings[0].unsqueeze(0), sentence_embeddings[1].unsqueeze(0))

            # Return similarity score (scale 0-1) * 100
            return max(0.0, float(similarity[0][0])) * 100 # Ensure score is >= 0

        except Exception as e:
            self.status_label.config(text=f"Error during similarity calculation: {e}")
            print(f"Error calculating transformer similarity: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback for debugging
            return 0.0 # Return 0 on error
    # --- End Transformer Logic ---


    # --- Main Comparison Methods ---
    def compare_sentences(self, sentence1, sentence2, method_name):
        """
        Compare two sentences using the specified method and return similarity score (0-100).
        """
        if method_name == "WordNet (Graph-based)" or method_name == "wordnet":
            return self._calculate_wordnet_similarity(sentence1, sentence2)
        elif TRANSFORMERS_AVAILABLE:
            # Ensure the correct model is loaded before calculating
            if self.loaded_model_name != method_name:
                if not self.load_similarity_model(method_name):
                    return 0.0 # Model loading failed
            # Now calculate using the loaded model
            return self._calculate_transformer_similarity(sentence1, sentence2)
        else:
            messagebox.showerror("Error", f"Method '{method_name}' requires the 'transformers' library, which is not installed.")
            return 0.0

    def compare_files(self, file1, file2, method_name):
        """Compare two files using the specified method and return similarity score (0-100)"""
        content1 = self.read_file_content(file1)
        content2 = self.read_file_content(file2)

        if content1 is None or content2 is None:
            self.status_label.config(text=f"Skipping comparison due to read error for {file1} or {file2}")
            return 0.0 # Error reading one or both files

        # Handle empty files
        if not content1.strip() or not content2.strip():
            # print(f"Warning: One or both files are empty ({file1}, {file2}). Similarity set to 0.")
            return 0.0 # Define similarity for empty files as 0

        # Simple comparison for now, consider chunking for very large files
        return self.compare_sentences(content1, content2, method_name)

    def find_similar_files(self, similarity_threshold=95.0, method_name="wordnet"):
        """Find files with similarity above the threshold using the specified method"""
        if not self.current_user:
             messagebox.showerror("Error", "Please login first.")
             return []

        files = self.get_user_files()
        if len(files) < 2:
             self.status_label.config(text="Need at least two files to compare.")
             return []

        similar_pairs = []
        total_comparisons = len(files) * (len(files) - 1) // 2
        comparisons_done = 0

        self.status_label.config(text=f"Starting comparison ({method_name})... 0/{total_comparisons}")
        self.root.update_idletasks()

        # Pre-load model if it's a transformer method and not already loaded
        if method_name != "wordnet" and TRANSFORMERS_AVAILABLE:
            if not self.load_similarity_model(method_name):
                self.status_label.config(text="Comparison cancelled: Model load failed.")
                return [] # Stop if model loading fails

        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                comparisons_done += 1
                # Update status periodically
                if comparisons_done % 5 == 0 or comparisons_done == total_comparisons:
                     self.status_label.config(text=f"Comparing... {comparisons_done}/{total_comparisons}")
                     self.root.update_idletasks()

                similarity = self.compare_files(files[i], files[j], method_name)

                # Check for calculation errors (indicated by negative values if we used that)
                # or handle None if compare_files could return it
                if similarity is None:
                     print(f"Warning: Comparison failed between {files[i]} and {files[j]}")
                     continue # Skip this pair

                if similarity >= similarity_threshold:
                    similar_pairs.append((files[i], files[j], similarity))

        self.status_label.config(text=f"Comparison complete. Found {len(similar_pairs)} pairs above {similarity_threshold}%.")
        return similar_pairs
    # --- End Main Comparison Methods ---


    # --- UI Setup and Handlers ---
    def setup_ui(self):
        """Set up the user interface"""
        self.root = tk.Tk()
        self.root.title("File Similarity Tool")
        self.root.geometry("800x650") # Increased height slightly for status bar

        # --- Style ---
        style = ttk.Style()
        style.theme_use('clam') # Or 'alt', 'default', 'classic'

        # --- Main Frame ---
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create a notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Login tab
        self.login_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.login_frame, text="Login/Register") # Combined for simplicity

        # Files tab
        self.files_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.files_frame, text="Manage Files")

        # Comparison tab
        self.comparison_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.comparison_frame, text="Compare Files")

        # --- Login/Register Frame Content ---
        auth_form = ttk.LabelFrame(self.login_frame, text="Authentication", padding="10")
        auth_form.pack(pady=20, padx=10, fill=tk.X)

        ttk.Label(auth_form, text="Username:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.auth_username = ttk.Entry(auth_form, width=30)
        self.auth_username.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(auth_form, text="Password:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.auth_password = ttk.Entry(auth_form, width=30, show="*")
        self.auth_password.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(auth_form, text="Confirm (for Register):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.auth_confirm = ttk.Entry(auth_form, width=30, show="*")
        self.auth_confirm.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        self.auth_confirm.config(state='disabled') # Initially disabled

        auth_form.columnconfigure(1, weight=1) # Make entry expand

        # Add a checkbox or radio buttons to choose Login/Register
        self.auth_mode = tk.StringVar(value="login")
        login_radio = ttk.Radiobutton(auth_form, text="Login", variable=self.auth_mode, value="login", command=self.toggle_auth_mode)
        register_radio = ttk.Radiobutton(auth_form, text="Register", variable=self.auth_mode, value="register", command=self.toggle_auth_mode)
        login_radio.grid(row=3, column=0, pady=10, sticky='w')
        register_radio.grid(row=3, column=1, pady=10, sticky='w')

        self.auth_button = ttk.Button(auth_form, text="Login", command=self.handle_auth)
        self.auth_button.grid(row=4, column=0, columnspan=2, pady=15)

        # --- Files Frame Content ---
        files_panel = ttk.LabelFrame(self.files_frame, text="Your Files", padding="10")
        files_panel.pack(fill=tk.BOTH, expand=True)

        self.files_label = ttk.Label(files_panel, text="Please login to view your files.")
        self.files_label.pack(pady=5, anchor='w')

        # Listbox with Scrollbar
        list_frame = ttk.Frame(files_panel)
        list_frame.pack(pady=5, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.files_listbox = tk.Listbox(list_frame, width=60, height=15, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.files_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Buttons Frame
        self.files_buttons_frame = ttk.Frame(files_panel)
        self.files_buttons_frame.pack(pady=10, fill=tk.X)
        ttk.Button(self.files_buttons_frame, text="Upload File(s)", command=self.handle_upload).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.files_buttons_frame, text="Refresh List", command=self.refresh_files).pack(side=tk.LEFT, padx=5)
        # Add Delete Button (Optional)
        # ttk.Button(self.files_buttons_frame, text="Delete Selected", command=self.handle_delete_file).pack(side=tk.LEFT, padx=5)


        # --- Comparison Frame Content ---
        compare_options_frame = ttk.LabelFrame(self.comparison_frame, text="Comparison Options", padding="10")
        compare_options_frame.pack(pady=10, fill=tk.X)

        # Method Selection
        ttk.Label(compare_options_frame, text="Similarity Method:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.method_var = tk.StringVar()
        self.method_dropdown = ttk.Combobox(compare_options_frame, textvariable=self.method_var, values=ALL_MODEL_NAMES, width=40, state='readonly')
        if ALL_MODEL_NAMES:
            self.method_dropdown.current(0) # Default to WordNet
        self.method_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Threshold
        ttk.Label(compare_options_frame, text="Similarity Threshold (%):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.threshold_var = tk.StringVar(value="95")
        self.threshold_entry = ttk.Entry(compare_options_frame, width=10, textvariable=self.threshold_var)
        self.threshold_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w") # Changed sticky

        compare_options_frame.columnconfigure(1, weight=1)

        # Compare Button
        ttk.Button(compare_options_frame, text="Find Similar Files", command=self.handle_comparison).grid(row=2, column=0, columnspan=2, pady=15)


        # Results Area
        results_frame = ttk.LabelFrame(self.comparison_frame, text="Results", padding="10")
        results_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.comparison_label = ttk.Label(results_frame, text="Results will appear here.", justify=tk.LEFT)
        self.comparison_label.pack(pady=5, anchor='w')

        # Results Listbox with Scrollbar
        results_list_frame = ttk.Frame(results_frame)
        results_list_frame.pack(pady=5, fill=tk.BOTH, expand=True)
        results_scrollbar = ttk.Scrollbar(results_list_frame, orient=tk.VERTICAL)
        self.comparison_listbox = tk.Listbox(results_list_frame, width=70, height=15, yscrollcommand=results_scrollbar.set)
        results_scrollbar.config(command=self.comparison_listbox.yview)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.comparison_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Status Bar ---
        self.status_label = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)


        # Disable tabs initially
        self.notebook.tab(1, state="disabled")
        self.notebook.tab(2, state="disabled")

    def toggle_auth_mode(self):
        """Updates UI based on login/register selection."""
        mode = self.auth_mode.get()
        if mode == "login":
             self.auth_confirm.config(state='disabled')
             self.auth_button.config(text="Login")
        else: # register
             self.auth_confirm.config(state='normal')
             self.auth_button.config(text="Register")

    def handle_auth(self):
        """Handle login or register button click"""
        mode = self.auth_mode.get()
        username = self.auth_username.get()
        password = self.auth_password.get()

        if not username or not password:
            messagebox.showerror("Error", "Please enter username and password.")
            return

        if mode == "login":
            self.handle_login(username, password)
        else: # register
            confirm = self.auth_confirm.get()
            if not confirm:
                messagebox.showerror("Error", "Please confirm your password.")
                return
            self.handle_register(username, password, confirm)

    def handle_login(self, username, password):
        """Handle login logic"""
        self.status_label.config(text="Attempting login...")
        self.root.update_idletasks()
        success, message = self.login_user(username, password)
        if success:
            messagebox.showinfo("Success", message)
            self.notebook.tab(1, state="normal") # Enable Files tab
            self.notebook.tab(2, state="normal") # Enable Comparison tab
            self.notebook.select(1) # Switch to Files tab
            self.refresh_files()
            self.files_label.config(text=f"Files for: {self.current_user}")
            self.status_label.config(text=f"Logged in as {self.current_user}.")
            # Clear fields after successful login
            self.auth_username.delete(0, tk.END)
            self.auth_password.delete(0, tk.END)
            self.auth_confirm.delete(0, tk.END)
        else:
            messagebox.showerror("Login Failed", message)
            self.status_label.config(text="Login failed.")

    def handle_register(self, username, password, confirm):
        """Handle register logic"""
        if password != confirm:
            messagebox.showerror("Error", "Passwords do not match.")
            return

        self.status_label.config(text="Attempting registration...")
        self.root.update_idletasks()
        success, message = self.register_user(username, password)
        if success:
            messagebox.showinfo("Success", message + "\nPlease login.")
            self.status_label.config(text="Registration successful. Please login.")
            # Switch back to login mode and clear fields
            self.auth_mode.set("login")
            self.toggle_auth_mode()
            self.auth_username.delete(0, tk.END)
            self.auth_password.delete(0, tk.END)
            self.auth_confirm.delete(0, tk.END)
            self.auth_username.focus() # Set focus back to username for login
        else:
            messagebox.showerror("Registration Failed", message)
            self.status_label.config(text="Registration failed.")

    def handle_upload(self):
        """Handle file upload"""
        if not self.current_user:
            messagebox.showerror("Error", "Please login first.")
            return

        file_paths = filedialog.askopenfilenames(
            title="Select text files to upload",
            # Added more common text-based types
            filetypes=[("Text Files", "*.txt *.py *.json *.csv *.md *.html *.xml"), ("All files", "*.*")]
        )

        if not file_paths:
            return

        uploaded_count = 0
        error_count = 0
        for file_path in file_paths:
            self.status_label.config(text=f"Uploading {Path(file_path).name}...")
            self.root.update_idletasks()
            success, message = self.upload_file(file_path)
            if success:
                 uploaded_count += 1
            else:
                 error_count += 1
                 messagebox.showerror("Upload Error", message) # Show error immediately

        final_message = f"Finished uploading. {uploaded_count} file(s) successful."
        if error_count > 0:
            final_message += f" {error_count} file(s) failed."
        self.status_label.config(text=final_message)
        self.refresh_files()

    def refresh_files(self):
        """Refresh the files list in the UI"""
        self.files_listbox.delete(0, tk.END)

        if not self.current_user:
            self.files_label.config(text="Please login to view your files.")
            self.notebook.tab(1, state="disabled") # Disable tabs if logged out
            self.notebook.tab(2, state="disabled")
            self.status_label.config(text="Logged out.")
            return

        # Ensure tabs are enabled if user is logged in
        self.notebook.tab(1, state="normal")
        self.notebook.tab(2, state="normal")
        self.files_label.config(text=f"Files for: {self.current_user}")

        try:
             files = self.get_user_files()
             if files:
                 for file in sorted(files): # Sort alphabetically
                     self.files_listbox.insert(tk.END, file)
             else:
                 self.files_listbox.insert(tk.END, "No files uploaded yet.")
             self.status_label.config(text=f"Loaded {len(files)} file(s) for {self.current_user}.")
        except Exception as e:
             messagebox.showerror("Error", f"Could not load file list: {e}")
             self.status_label.config(text="Error loading file list.")


    def handle_comparison(self):
        """Handle file comparison button click"""
        if not self.current_user:
            messagebox.showerror("Error", "Please login first.")
            return

        method_name = self.method_var.get()
        if not method_name:
            messagebox.showerror("Error", "Please select a similarity method.")
            return

        # Special handling for the display name vs internal name
        if method_name == "WordNet (Graph-based)":
            internal_method_name = "wordnet"
        else:
            internal_method_name = method_name

        if internal_method_name != "wordnet" and not TRANSFORMERS_AVAILABLE:
             messagebox.showerror("Error", f"Method '{method_name}' requires the 'transformers' library, which is not installed.")
             return


        try:
            threshold = float(self.threshold_var.get())
            if not (0 <= threshold <= 100):
                 raise ValueError("Threshold must be between 0 and 100.")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid threshold: {e}\nPlease enter a number between 0 and 100.")
            return

        self.comparison_listbox.delete(0, tk.END)
        self.comparison_label.config(text="Comparison in progress...")
        self.root.update_idletasks() # Show "in progress" message

        # Perform comparison
        try:
            similar_pairs = self.find_similar_files(threshold, internal_method_name)
        except Exception as e:
             # Catch unexpected errors during the comparison loop
             messagebox.showerror("Comparison Error", f"An error occurred during comparison: {e}")
             self.status_label.config(text="Comparison failed.")
             import traceback
             traceback.print_exc()
             return


        # Update results display
        if not similar_pairs:
            self.comparison_label.config(text=f"No similar file pairs found above {threshold}% using {method_name}.")
            self.comparison_listbox.insert(tk.END, "No results.")
        else:
            self.comparison_label.config(text=f"Found {len(similar_pairs)} similar file pairs (>= {threshold}%) using {method_name}:")
            # Sort pairs by similarity score (descending)
            similar_pairs.sort(key=lambda x: x[2], reverse=True)
            for file1, file2, similarity in similar_pairs:
                self.comparison_listbox.insert(tk.END, f"{similarity:.2f}% : {file1} <-> {file2}")
        # Final status update is handled within find_similar_files


    def run(self):
        """Run the application"""
        self.root.mainloop()

# --- End UI Setup and Handlers ---

if __name__ == "__main__":
    # --- Dependency Check ---
    missing_libs = []
    try:
        import nltk
    except ImportError:
        missing_libs.append("nltk")
    try:
        import numpy
    except ImportError:
        missing_libs.append("numpy")
    # try: import matplotlib except ImportError: missing_libs.append("matplotlib") # Optional
    # try: import networkx except ImportError: missing_libs.append("networkx") # Optional
    try:
        import pandas
    except ImportError:
        missing_libs.append("pandas")
    try:
        import sklearn
    except ImportError:
        missing_libs.append("scikit-learn")
    # Check transformers/torch only if they were expected to be available
    if TRANSFORMERS_AVAILABLE:
         pass # Already checked during import
    elif "transformers" not in str(ImportError): # Check if the initial import failed specifically for transformers
         try:
             import torch
         except ImportError:
              missing_libs.append("torch")
         try:
              import transformers
         except ImportError:
              missing_libs.append("transformers")


    if missing_libs:
        print("="*50)
        print("ERROR: Missing required Python libraries!")
        print("Please install them using pip:")
        print(f"  pip install {' '.join(missing_libs)}")
        if "nltk" in missing_libs:
             print("\nAfter installing nltk, you might need to run Python and execute:")
             print("import nltk")
             print("nltk.download('punkt')")
             print("nltk.download('stopwords')")
             print("nltk.download('wordnet')")
             print("nltk.download('averaged_perceptron_tagger')")
        print("="*50)
        # Optionally exit or show a GUI error message
        root = tk.Tk()
        root.withdraw() # Hide main window
        messagebox.showerror("Dependency Error", "Missing required libraries:\n" + "\n".join(missing_libs) + "\nPlease install them (see console output) and restart the tool.")
        # exit(1) # Or just let the app potentially crash later
    else:
        print("All core dependencies seem to be installed.")


    # --- Run Application ---
    app = FileSimilarityTool()
    app.run()