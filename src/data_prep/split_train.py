import pandas

CSV_PATH = "../../data/book-dataset/book30-listing-train.csv"

# Read in CSV file. Note: CSV has weird encoding, so we need to specify it.
df = pandas.read_csv(CSV_PATH, encoding="latin")

# Add names for each column
df.columns = ["asin", "filename", "image_url", "title", "author", "cat_id", "category"]


# Split into train and val
train = df.sample(frac=0.8, random_state=200)
val = df.drop(train.index)

# Save to CSV
train.to_csv("../../data/book-dataset/book30-listing-train-train.csv", index=False)
val.to_csv("../../data/book-dataset/book30-listing-train-val.csv", index=False)