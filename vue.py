import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from randomForest import RandomForest
from CART import CART

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Algorithmes de Machine Learning")
        self.geometry("600x400")

        self.label = tk.Label(self, text="Choisissez un fichier CSV:")
        self.label.pack(pady=10)

        self.button = tk.Button(self, text="Parcourir", command=self.load_file)
        self.button.pack(pady=10)

        self.algorithm_var = tk.StringVar(value="RandomForest")
        self.radio_rf = tk.Radiobutton(self, text="RandomForest", variable=self.algorithm_var, value="RandomForest")
        self.radio_rf.pack(pady=5)
        self.radio_cart = tk.Radiobutton(self, text="CART", variable=self.algorithm_var, value="CART")
        self.radio_cart.pack(pady=5)

        self.run_button = tk.Button(self, text="Exécuter", command=self.run_algorithm)
        self.run_button.pack(pady=20)

        self.result_label = tk.Label(self, text="")
        self.result_label.pack(pady=10)

        self.data = None

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.data = pd.read_csv(file_path)
            messagebox.showinfo("Info", "Fichier chargé avec succès!")

    def run_algorithm(self):
        if self.data is None:
            messagebox.showerror("Erreur", "Veuillez charger un fichier CSV d'abord.")
            return

        algorithm = self.algorithm_var.get()
        if algorithm == "RandomForest":
            model = RandomForest(self.data, n_trees=10, n_features=self.data.shape[1] - 1)
            model.set_features(self.data.columns.tolist())
            model.fit()
            score = model.score(self.data)
        elif algorithm == "CART":
            model = CART(self.data)
            target = self.data.columns[-1]
            categorical = [col for col in self.data.columns if self.data[col].dtype == 'object']
            numerical = [col for col in self.data.columns if self.data[col].dtype != 'object']
            model.fit(target, categorical, numerical)
            model.predict(self.data)
            score = model.evaluate()

        self.result_label.config(text=f"Score: {score:.2f}")
        messagebox.showinfo("Résultat", f"Score: {score:.2f}")

