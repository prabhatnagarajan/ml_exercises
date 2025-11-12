import pandas as pd
import os
	
def download_iris():
	url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
	cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
	if not os.path.exists("datasets/iris.csv"):
		print("Downloading Iris dataset...")
		df = pd.read_csv(url, names=cols)
		if not os.path.exists("datasets"):
			os.makedirs("datasets")
		df.to_csv("datasets/iris.csv", index=False)
		print("Iris dataset downloaded and saved to datasets/iris.csv")
	else:
		print("Iris dataset already exists at datasets/iris.csv")

def iris_train_test_split(test_size=0.2, random_state=42):
	download_iris()
	df = pd.read_csv("datasets/iris.csv")
	df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)  # Shuffle the dataset
	split_index = int(len(df) * (1 - test_size))
	train_df = df[:split_index]
	test_df = df[split_index:]
	train_labels = train_df.pop("class")
	test_labels = test_df.pop("class") 
	return train_df, train_labels, test_df, test_labels