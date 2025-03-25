import os
import hashlib
import json
import nltk
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from nltk.corpus import stopwords, wordnet as wn
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class FileSimilarityTool:
    def __init__(self):
        self.users = {}
        self.user_files = {}
        self.current_user = None
        self.data_dir = Path("data")
        self.users_file = self.data_dir / "users.json"
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
        self.user_files[username] = []
        return True, "User registered successfully"
    
    def login_user(self, username, password):
        """Login an existing user"""
        if username not in self.users:
            return False, "User does not exist"
        
        if self.users[username]["password_hash"] != self.hash_password(password):
            return False, "Incorrect password"
        
        self.current_user = username
        return True, "Login successful"
    
    def save_users(self):
        """Save user data to the JSON file"""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f)
    
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
            
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as src_file:
                content = src_file.read()
            
            # Save the file in the user's directory
            with open(dest_path, 'w', encoding='utf-8') as dest_file:
                dest_file.write(content)
            
            # Update user's files list
            if file_name not in self.users[self.current_user]["files"]:
                self.users[self.current_user]["files"].append(file_name)
                self.save_users()
            
            return True, f"File {file_name} uploaded successfully"
        except Exception as e:
            return False, f"Error uploading file: {str(e)}"
    
    def get_user_files(self):
        """Get the list of files for the current user"""
        if not self.current_user:
            return []
        return self.users[self.current_user]["files"]
    
    def read_file_content(self, filename):
        """Read the content of a file"""
        if not self.current_user:
            return None
        
        file_path = self.user_directory(self.current_user) / filename
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None
    
    # This function includes the semantic similarity code from the provided example
    def create_semantic_graph(self, sentence):
        """Creates a semantic graph with weighted relationships."""
        # Weights for different semantic relationships
        WEIGHTS = {
            'exact': 1.0,
            'synonym': 0.9,
            'hypernym': 0.8,
            'hyponym': 0.8,
            'meronym': 0.6,
            'holonym': 0.6
        }

        G = nx.DiGraph()
        words = nltk.word_tokenize(sentence.lower())
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words]
        tagged_words = nltk.pos_tag(filtered_words)

        # Track all nodes and their relationships
        word_synsets = {}

        # Track node types for visualization
        node_types = {}

        for word, pos in tagged_words:
            if pos[0] in ['N', 'V', 'J']:  # Consider nouns, verbs, and adjectives
                word_node = f"{word}_{pos}"
                G.add_node(word_node, type='word')
                node_types[word_node] = 'word'

                # Get all synsets for the word
                synsets = wn.synsets(word)
                if not synsets:
                    continue

                # Add synset nodes and relationships
                for synset in synsets:
                    synset_node = str(synset.name())
                    G.add_node(synset_node, type='synset')
                    node_types[synset_node] = 'synset'
                    G.add_edge(word_node, synset_node, weight=WEIGHTS['exact'])

                    # Store for later similarity calculation
                    if word_node not in word_synsets:
                        word_synsets[word_node] = []
                    word_synsets[word_node].append(synset)

                    # Add hypernyms
                    for hypernym in synset.hypernyms():
                        hyp_node = str(hypernym.name())
                        G.add_node(hyp_node, type='hypernym')
                        node_types[hyp_node] = 'hypernym'
                        G.add_edge(synset_node, hyp_node, weight=WEIGHTS['hypernym'])

                    # Add hyponyms
                    for hyponym in synset.hyponyms():
                        hypo_node = str(hyponym.name())
                        G.add_node(hypo_node, type='hyponym')
                        node_types[hypo_node] = 'hyponym'
                        G.add_edge(synset_node, hypo_node, weight=WEIGHTS['hyponym'])

                    # Add meronyms
                    for meronym in synset.part_meronyms() + synset.member_meronyms():
                        mero_node = str(meronym.name())
                        G.add_node(mero_node, type='meronym')
                        node_types[mero_node] = 'meronym'
                        G.add_edge(synset_node, mero_node, weight=WEIGHTS['meronym'])

                    # Add holonyms
                    for holonym in synset.part_holonyms() + synset.member_holonyms():
                        holo_node = str(holonym.name())
                        G.add_node(holo_node, type='holonym')
                        node_types[holo_node] = 'holonym'
                        G.add_edge(synset_node, holo_node, weight=WEIGHTS['holonym'])

        return G, word_synsets, node_types

    def calculate_word_similarity(self, synsets1, synsets2):
        """Calculate similarity between two sets of synsets."""
        max_sim = 0.0

        for syn1 in synsets1:
            for syn2 in synsets2:
                # Check for exact match or synonym
                if syn1 == syn2:
                    return 1.0

                # Check lemma overlap (synonyms)
                lemmas1 = set(l.name() for l in syn1.lemmas())
                lemmas2 = set(l.name() for l in syn2.lemmas())
                if lemmas1.intersection(lemmas2):
                    return 0.9

                # Calculate path similarity
                path_sim = syn1.path_similarity(syn2) or 0.0

                # Check hypernym/hyponym relationship
                if syn2 in syn1.hypernyms() or syn1 in syn2.hypernyms():
                    path_sim = max(path_sim, 0.8)

                # Use shortest path if available
                max_sim = max(max_sim, path_sim)

        return max_sim

    def compare_sentences(self, sentence1, sentence2, show_graphs=False):
        """Compare two sentences using semantic similarity and visualize their graphs."""
        # Creating semantic graphs for both sentences
        graph1, words1, types1 = self.create_semantic_graph(sentence1)
        graph2, words2, types2 = self.create_semantic_graph(sentence2)

        # Calculating similarities...
        similarities = []
        for word1, synsets1 in words1.items():
            word_sims = []
            for word2, synsets2 in words2.items():
                sim = self.calculate_word_similarity(synsets1, synsets2)
                word_sims.append(sim)
            if word_sims:
                similarities.append(max(word_sims))

        if not similarities:
            return 0.0

        # Calculating overall similarity
        avg_similarity = sum(similarities) / len(similarities)
        coverage = len(similarities) / max(len(words1), len(words2))

        if coverage > 0.8 and avg_similarity > 0.7:
            final_score = min(1.0, avg_similarity * 1.2)
        else:
            final_score = avg_similarity

        return final_score * 100
    
    def compare_files(self, file1, file2):
        """Compare two files and return similarity score"""
        content1 = self.read_file_content(file1)
        content2 = self.read_file_content(file2)
        
        if content1 is None or content2 is None:
            return 0.0
        
        # For large files, we might want to chunk the content
        # but for simplicity, we'll compare the entire contents
        return self.compare_sentences(content1, content2, show_graphs=False)
    
    def find_similar_files(self, similarity_threshold=95.0):
        """Find files with similarity above the threshold"""
        files = self.get_user_files()
        similar_pairs = []
        
        for i in range(len(files)):
            for j in range(i+1, len(files)):
                similarity = self.compare_files(files[i], files[j])
                if similarity >= similarity_threshold:
                    similar_pairs.append((files[i], files[j], similarity))
        
        return similar_pairs
    
    def setup_ui(self):
        """Set up the user interface"""
        self.root = tk.Tk()
        self.root.title("File Similarity Tool")
        self.root.geometry("800x600")
        
        # Create a notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Login tab
        self.login_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.login_frame, text="Login")
        
        # Register tab
        self.register_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.register_frame, text="Register")
        
        # Files tab
        self.files_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.files_frame, text="Files")
        
        # Comparison tab
        self.comparison_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.comparison_frame, text="Comparison")
        
        # Set up login form
        ttk.Label(self.login_frame, text="Username:").grid(row=0, column=0, padx=10, pady=10)
        self.login_username = ttk.Entry(self.login_frame, width=30)
        self.login_username.grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(self.login_frame, text="Password:").grid(row=1, column=0, padx=10, pady=10)
        self.login_password = ttk.Entry(self.login_frame, width=30, show="*")
        self.login_password.grid(row=1, column=1, padx=10, pady=10)
        
        ttk.Button(self.login_frame, text="Login", command=self.handle_login).grid(row=2, column=0, columnspan=2, pady=20)
        
        # Set up register form
        ttk.Label(self.register_frame, text="Username:").grid(row=0, column=0, padx=10, pady=10)
        self.register_username = ttk.Entry(self.register_frame, width=30)
        self.register_username.grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(self.register_frame, text="Password:").grid(row=1, column=0, padx=10, pady=10)
        self.register_password = ttk.Entry(self.register_frame, width=30, show="*")
        self.register_password.grid(row=1, column=1, padx=10, pady=10)
        
        ttk.Label(self.register_frame, text="Confirm Password:").grid(row=2, column=0, padx=10, pady=10)
        self.register_confirm = ttk.Entry(self.register_frame, width=30, show="*")
        self.register_confirm.grid(row=2, column=1, padx=10, pady=10)
        
        ttk.Button(self.register_frame, text="Register", command=self.handle_register).grid(row=3, column=0, columnspan=2, pady=20)
        
        # Set up files tab
        self.files_label = ttk.Label(self.files_frame, text="Please login to view your files")
        self.files_label.pack(pady=10)
        
        self.files_listbox = tk.Listbox(self.files_frame, width=60, height=15)
        self.files_listbox.pack(pady=10, fill=tk.BOTH, expand=True)
        
        self.files_buttons_frame = ttk.Frame(self.files_frame)
        self.files_buttons_frame.pack(pady=10, fill=tk.X)
        
        ttk.Button(self.files_buttons_frame, text="Upload File", command=self.handle_upload).pack(side=tk.LEFT, padx=10)
        ttk.Button(self.files_buttons_frame, text="Refresh", command=self.refresh_files).pack(side=tk.LEFT, padx=10)
        
        # Set up comparison tab
        self.comparison_label = ttk.Label(self.comparison_frame, text="Results will appear here", justify=tk.LEFT)
        self.comparison_label.pack(pady=10)
        
        self.comparison_listbox = tk.Listbox(self.comparison_frame, width=60, height=15)
        self.comparison_listbox.pack(pady=10, fill=tk.BOTH, expand=True)
        
        self.threshold_frame = ttk.Frame(self.comparison_frame)
        self.threshold_frame.pack(pady=10, fill=tk.X)
        
        ttk.Label(self.threshold_frame, text="Similarity Threshold (%):").pack(side=tk.LEFT, padx=10)
        self.threshold_var = tk.StringVar(value="95")
        self.threshold_entry = ttk.Entry(self.threshold_frame, width=10, textvariable=self.threshold_var)
        self.threshold_entry.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(self.threshold_frame, text="Compare Files", command=self.handle_comparison).pack(side=tk.LEFT, padx=10)
        
        # Disable tabs initially
        self.notebook.tab(2, state="disabled")
        self.notebook.tab(3, state="disabled")
    
    def handle_login(self):
        """Handle login button click"""
        username = self.login_username.get()
        password = self.login_password.get()
        
        if not username or not password:
            messagebox.showerror("Error", "Please enter both username and password")
            return
        
        success, message = self.login_user(username, password)
        if success:
            messagebox.showinfo("Success", message)
            self.notebook.tab(2, state="normal")
            self.notebook.tab(3, state="normal")
            self.notebook.select(2)
            self.refresh_files()
            self.files_label.config(text=f"Files for {username}")
        else:
            messagebox.showerror("Error", message)
    
    def handle_register(self):
        """Handle register button click"""
        username = self.register_username.get()
        password = self.register_password.get()
        confirm = self.register_confirm.get()
        
        if not username or not password or not confirm:
            messagebox.showerror("Error", "Please fill in all fields")
            return
        
        if password != confirm:
            messagebox.showerror("Error", "Passwords do not match")
            return
        
        success, message = self.register_user(username, password)
        if success:
            messagebox.showinfo("Success", message)
            self.notebook.select(0)
            self.register_username.delete(0, tk.END)
            self.register_password.delete(0, tk.END)
            self.register_confirm.delete(0, tk.END)
        else:
            messagebox.showerror("Error", message)
    
    def handle_upload(self):
        """Handle file upload"""
        if not self.current_user:
            messagebox.showerror("Error", "Please login first")
            return
        
        file_paths = filedialog.askopenfilenames(
            title="Select text files",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not file_paths:
            return
        
        for file_path in file_paths:
            success, message = self.upload_file(file_path)
            if not success:
                messagebox.showerror("Error", message)
        
        self.refresh_files()
    
    def refresh_files(self):
        """Refresh the files list"""
        self.files_listbox.delete(0, tk.END)
        
        if not self.current_user:
            self.files_label.config(text="Please login to view your files")
            return
        
        files = self.get_user_files()
        for file in files:
            self.files_listbox.insert(tk.END, file)
    
    def handle_comparison(self):
        """Handle file comparison"""
        if not self.current_user:
            messagebox.showerror("Error", "Please login first")
            return
        
        try:
            threshold = float(self.threshold_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid threshold")
            return
        
        self.comparison_listbox.delete(0, tk.END)
        
        similar_pairs = self.find_similar_files(threshold)
        
        if not similar_pairs:
            self.comparison_label.config(text="No similar files found")
            return
        
        self.comparison_label.config(text=f"Found {len(similar_pairs)} similar file pairs")
        
        for file1, file2, similarity in similar_pairs:
            self.comparison_listbox.insert(tk.END, f"{file1} <-> {file2}: {similarity:.2f}% similar")

    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = FileSimilarityTool()
    app.run()