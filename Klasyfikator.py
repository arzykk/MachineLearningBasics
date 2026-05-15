import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

plt.scatter(X[:, 0], X[:, 1], c=y, s=10)
plt.title("make_moons dataset")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


depths = [2, 10, 100]

for criterion in ["gini", "entropy"]:
    print(f"\nCriterion: {criterion}")
    for d in depths:
        clf = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=d,
            random_state=42
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"depth={d}, acc={accuracy_score(y_test, y_pred):.4f}")

print()

n_trees = [1, 10, 100, 500]

for n in n_trees:
    rf = RandomForestClassifier(
        n_estimators=n,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print(f"trees={n}, acc={accuracy_score(y_test, y_pred):.4f}")

print()

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

svm = SVC(probability=True)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("Logistic Regression:", accuracy_score(y_test, y_pred_lr))
print("SVM:", accuracy_score(y_test, y_pred_svm))


rf = RandomForestClassifier(n_estimators=100, random_state=42)

voting = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('rf', rf),
        ('svm', svm)
    ],
    voting='soft'
)

voting.fit(X_train, y_train)
y_pred_vote = voting.predict(X_test)

print("VotingClassifier:", accuracy_score(y_test, y_pred_vote))
