import pandas as pd # veri okumaya yardımcı olur
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = load_iris()

X = pd.DataFrame(data.data)  # iris data
y = pd.DataFrame(data.target)  # gerçek(hedef) değerler

# dataset parçalanıp test datası oluşturuluyor
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.4)


def training_model():
    model = DecisionTreeClassifier()
    trained_model = model.fit(X_train, y_train) # modeli eğitir
    return trained_model
