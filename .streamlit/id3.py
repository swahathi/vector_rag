import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"

try:
    df = pd.read_excel(url)

except Exception as e:
    print("Could not download dataset. Using synthetic sample instead:", e)

    rng = np.random.RandomState(1)
    n = 2000

    df = pd.DataFrame({
        "CustomerID": rng.randint(1000, 1200, n),
        "InvoiceNo": rng.randint(20000, 25000, n),
        "Quantity": rng.randint(1, 20, n),
        "UnitPrice": np.round(rng.uniform(1, 50, n), 2),
        "InvoiceDate": pd.date_range("2011-01-01", periods=n, freq="h")
    })


# 2. Clean data
df = df.dropna(subset=["CustomerID"]) # customer id col ethelum value missing aahnel delete 
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]


# 3. Create RFM features for each customer
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

customer = df.groupby("CustomerID").agg( #same customer id ullathellam onnakkunnu
    Recency=("InvoiceDate",
             lambda x: (snapshot_date - x.max()).days),

    Frequency=("InvoiceNo", "nunique"), #unique dont count duplicate invoice numbers

    Monetary=("TotalPrice", "sum")
).reset_index()


# 4. Create High Value / Low Value labels
median_spend = customer["Monetary"].median() #median of all monetary

customer["Segment"] = np.where( # segment is a new column
    customer["Monetary"] > median_spend,
    "High Value",
    "Low Value"
)


# 5. Input and output
X = customer[["Recency", "Frequency", "Monetary"]]
y = customer["Segment"]


# 6. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 7. Create and train ID3 Decision Tree
id3 = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    random_state=42
)

id3.fit(X_train, y_train)


# 8. Accuracy
print("Train accuracy:", id3.score(X_train, y_train))
print("Test accuracy :", id3.score(X_test, y_test))


# 9. Feature importance
importance = pd.Series(
    id3.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature importance:\n", importance)


# 10. Visualise the tree
plt.figure(figsize=(14, 7))

plot_tree(
    id3,
    feature_names=X.columns,
    class_names=id3.classes_,
    filled=True,
    fontsize=9
)

plt.title("ID3 (entropy) Decision Tree - Customer Segmentation")
plt.show()