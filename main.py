import pandas as pd
from sklearn.model_selection import train_test_split

# synthetic data
data = {
    "age": [25, 32, 47, 51, 62, 23, 38, 44],
    "salary": [2500, 3200, 4700, 5100, 6200, 2300, 3800, 4400],
    "purchased": [0, 1, 1, 1, 1, 0, 0, 1]  
}

df = pd.DataFrame(data)

X = df[["age", "salary"]] 
y = df["purchased"]        

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,       
    random_state=42,      
    stratify=y            
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
