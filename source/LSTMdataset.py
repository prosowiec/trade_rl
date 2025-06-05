import tensorflow as tf
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class WindowDataset(Dataset):
    def __init__(self, data, window_generator):
        """
        data: Oryginalne dane do generowania sekwencji (np. np.array lub torch.tensor)
        window_generator: Obiekt klasy WindowGenerator, który ma metodę split_window
        """
        self.data = torch.tensor(data, dtype=torch.float32)  # konwersja na tensor
        self.window_generator = window_generator
        self.total_window_size = window_generator.total_window_size

    def __len__(self):
        return len(self.data) - self.total_window_size + 1

    def __getitem__(self, idx):
        window = self.data[idx:idx + self.total_window_size]  # tworzenie okna
        window = window.unsqueeze(0)  # Dodajemy wymiar batchu
        self.inputs, self.labels = self.window_generator.split_window(window)
        return self.inputs.squeeze(0), self.labels.squeeze(0)  # Usuwamy wymiar batchu, by były [time, features]
        


class WindowGenerator():
	def __init__(self, input_width, label_width, shift,
				 train_df, val_df, test_df,
				 label_columns=None):
		# Store the raw data.
		self.train_df = train_df
		self.val_df = val_df
		self.test_df = test_df

		# Work out the label column indices.
		self.label_columns = label_columns
		if label_columns is not None:
			self.label_columns_indices = {name: i for i, name in
										  enumerate(label_columns)}
		self.column_indices = {name: i for i, name in
							   enumerate(train_df.columns)}

		# Work out the window parameters.
		self.input_width = input_width
		self.label_width = label_width
		self.shift = shift

		self.total_window_size = input_width + shift

		self.input_slice = slice(0, input_width)
		self.input_indices = np.arange(self.total_window_size)[self.input_slice]

		self.label_start = self.total_window_size - self.label_width
		self.labels_slice = slice(self.label_start, None)
		self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

	def __repr__(self):
		return '\n'.join([
			f'Total window size: {self.total_window_size}',
			f'Input indices: {self.input_indices}',
			f'Label indices: {self.label_indices}',
			f'Label column name(s): {self.label_columns}'])

	def split_window(self, features):
		inputs = features[:, self.input_slice, :]
		labels = features[:, self.labels_slice, :]
		if self.label_columns is not None:
			labels = torch.stack(
				[labels[:, :, self.column_indices[name]] for name in self.label_columns],
				axis=-1)

		# Slicing doesn't preserve static shape information, so set the shapes
		# manually. This way the `tf.data.Datasets` are easier to inspect.
		#inputs.set_shape([None, self.input_width, None])
		#labels.set_shape([None, self.label_width, None])

		return inputs, labels

	def plot(self, predictions=None, plot_col='Close', max_subplots=10):
		inputs, labels = self.example
		plt.figure(figsize=(12, 8))
		plot_col_index = self.column_indices[plot_col]
		max_n = min(max_subplots, len(inputs))
		for n in range(max_n):
			plt.subplot(max_n, 1, n + 1)
			plt.ylabel(f'{plot_col} [normed]')
			plt.plot(self.input_indices, inputs[n, :, plot_col_index],
					 label='Inputs', marker='.', zorder=-10)

			if self.label_columns:
				label_col_index = self.label_columns_indices.get(plot_col, None)
			else:
				label_col_index = plot_col_index

			if label_col_index is None:
				continue

			plt.scatter(self.label_indices, labels[n, :, label_col_index],
						edgecolors='k', label='Labels', c='#2ca02c', s=64)
			if predictions is not None:
				#predictions = model(inputs)
				#predictions = predictions.detach().numpy()
				plt.scatter(self.label_indices, predictions[n, :, label_col_index],
							marker='X', edgecolors='k', label='Predictions',
							c='#ff7f0e', s=64)

			if n == 0:
				plt.legend()

		plt.xlabel('Time [h]')
  
	def make_dataset(self, data):
		"""
		Tworzy dataset na podstawie danych i okna z klasy WindowGenerator.
		"""
		dataset = WindowDataset(data, self)  # Tworzymy dataset z danymi i WindowGenerator
		return dataset

