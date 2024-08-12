import os
import pandas as pd
from itertools import combinations
from collections import defaultdict
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog, QMessageBox

class AssociationRuleMiningApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Association Rule Mining")
        self.setGeometry(100, 100, 800, 600)

        # Central Widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(layout)

        # File Selection
        file_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(file_layout)

        file_label = QtWidgets.QLabel("Select CSV file:")
        file_layout.addWidget(file_label)

        self.file_entry = QtWidgets.QLineEdit()
        file_layout.addWidget(self.file_entry)

        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_button)

        # Parameters
        parameters_frame = QtWidgets.QGroupBox("Parameters")
        layout.addWidget(parameters_frame)

        parameters_layout = QtWidgets.QFormLayout()
        parameters_frame.setLayout(parameters_layout)

        self.num_records_entry = QtWidgets.QLineEdit("100")
        parameters_layout.addRow("Number of Records:", self.num_records_entry)

        self.min_support_entry = QtWidgets.QLineEdit("5")
        parameters_layout.addRow("Minimum Support:", self.min_support_entry)

        self.min_confidence_entry = QtWidgets.QLineEdit("50")
        parameters_layout.addRow("Minimum Confidence:", self.min_confidence_entry)

        # Mine Association Rules Button
        mine_button = QtWidgets.QPushButton("Mine Association Rules")
        mine_button.clicked.connect(self.mine_association_rules)
        layout.addWidget(mine_button)

        # Output
        output_label = QtWidgets.QLabel("Output:")
        layout.addWidget(output_label)

        self.output_text = QtWidgets.QTextEdit()
        layout.addWidget(self.output_text)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV file", "", "CSV files (*.csv)")
        if file_path:
            self.file_entry.setText(file_path)

    def load_data(self, file_path, num_records):
        # Read CSV file and select the first num_records
        df = pd.read_csv(file_path).head(num_records)

        # Group by 'TransactionNo' and aggregate 'Items' as a list
        #Initializes an empty defaultdict named grouped_data to store grouped transactional data where the keys are transaction numbers (TransactionNo) and the values are lists of items.
        grouped_data = defaultdict(list)
        for index, row in df.iterrows():
            grouped_data[row['TransactionNo']].extend(row['Items'].split(','))

        return grouped_data

    def calculate_support_count(self, dataset):
        item_counts = defaultdict(int)
        for items in dataset.values():
            unique_items = set()  # Create an empty set to store unique items in each transaction
            for item in items:
                if item not in unique_items:  # Check if the item has not been counted in this transaction yet
                    item_counts[item] += 1  # Increment the support count for the item
                    unique_items.add(item)  # Add the item to the set to mark it as counted in this transaction
        return item_counts

    def calculate_combinations(self, dataset):
        combination_counts = defaultdict(int)
        for items in dataset.values():
            for r in range(2, len(items) + 1):
                for combination in combinations(items, r):
                    combination_counts[tuple(sorted(combination))] += 1
        return combination_counts

    def filter_combinations(self, combination_counts, min_support):
        return {comb: count for comb, count in combination_counts.items() if count >= min_support}

    def generate_association_rules(self, filtered_combinations, item_counts, min_confidence):
        association_rules = []
        for items, count in filtered_combinations.items():
            for item in items:
                antecedent = ', '.join(sorted(set(items) - {item}))
                antecedent_count = sum(item_counts[ant_item] for ant_item in antecedent.split(', '))
                if antecedent_count == 0:
                    continue  # Skip calculation if antecedent count is zero
                confidence = (count / antecedent_count) * 100
                if confidence >= min_confidence:
                    rule = f"If {' and '.join(antecedent.split(', '))} then {item} - Confidence: {confidence:.2f}%"
                    association_rules.append(rule)
        return association_rules

    def mine_association_rules(self):
        file_path = self.file_entry.text()
        if not os.path.exists(file_path):
            QMessageBox.critical(self, "Error", "File not found!")
            return

        num_records = int(self.num_records_entry.text())
        min_support = int(self.min_support_entry.text())
        min_confidence = int(self.min_confidence_entry.text())

        # Load data
        dataset = self.load_data(file_path, num_records)

        # Calculate support count for each item
        item_counts = self.calculate_support_count(dataset)

        # Calculate combinations and their counts
        combination_counts = self.calculate_combinations(dataset)

        # Filter combinations based on minimum support count
        filtered_combinations = self.filter_combinations(combination_counts, min_support)

        # Generate association rules
        association_rules = self.generate_association_rules(filtered_combinations, item_counts, min_confidence)

        # Display results in the output text box
        self.output_text.clear()
        self.output_text.append("Frequent Item Sets:")
        for items, count in filtered_combinations.items():
            self.output_text.append(f"{items}: {count}")

        self.output_text.append("\nStrong Association Rules:")
        for rule in association_rules:
            self.output_text.append(rule)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    main_window = AssociationRuleMiningApp()
    main_window.show()
    app.exec_()
