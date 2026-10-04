from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

CSV_PATH = Path(__file__).with_name("products.csv")
LABEL_COLUMN = "product"

def train_recommender(csv_path=CSV_PATH):
	"""Train a product classifier and return its model, features, and accuracy."""
	products = pd.read_csv(csv_path)

	if LABEL_COLUMN not in products.columns:
		raise ValueError(f"CSV must contain a '{LABEL_COLUMN}' column.")

	features = products.drop(columns=[LABEL_COLUMN])
	labels = products[LABEL_COLUMN]

	if features.empty:
		raise ValueError("CSV must contain at least one feature column.")
	if products.isna().any().any():
		raise ValueError("CSV must not contain missing values.")

	non_numeric = features.select_dtypes(exclude=[np.number]).columns.tolist()
	if non_numeric:
		raise ValueError(f"Feature columns must be numeric: {non_numeric}")

	X_train, X_test, y_train, y_test = train_test_split(
		features, labels, test_size=0.2, random_state=42
	)
	model = KNeighborsClassifier(n_neighbors=min(5, len(X_train)))
	model.fit(X_train, y_train)
	accuracy = model.score(X_test, y_test)

	return model, features.columns.tolist(), accuracy


def recommend(product_features, model, feature_columns):
	"""Predict a product from a dict or ordered sequence of feature values."""
	if isinstance(product_features, (dict, pd.Series)):
		missing = [name for name in feature_columns if name not in product_features]
		if missing:
			raise ValueError(f"Missing feature values: {missing}")
		values = [product_features[name] for name in feature_columns]
	else:
		values = product_features

	values = np.asarray(values, dtype=float)
	if values.ndim != 1 or len(values) != len(feature_columns):
		raise ValueError(
			f"Expected {len(feature_columns)} values in this order: {feature_columns}"
		)

	return model.predict(values.reshape(1, -1))[0]


if __name__ == "__main__":
	model, feature_columns, accuracy = train_recommender()
	print(f"Model accuracy: {accuracy:.2%}")
