import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("C:/Users/oguzhan/Downloads/Diseases_Symptoms.csv")

df['Features'] = df['Symptoms'] + ", " + df['Treatments'].fillna('')

X = df['Features']  # Birleştirilmiş metin özellikleri
y = df['Contagious']  # Tahmin edilecek hedef (bulaşıcı mı?)

vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '), max_features=500)
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.25, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Doğruluğu: %{accuracy*100:.2f}")
print("\nDetaylı Rapor:")
print(classification_report(y_test, y_pred, target_names=['Bulaşıcı Değil', 'Bulaşıcı']))

yeni_veri = ["cough, fever, antibiotics"]  
yeni_veri_vector = vectorizer.transform(yeni_veri)
tahmin = model.predict(yeni_veri_vector)
print(f"\nYeni Örnek Tahmini: {'Bulaşıcı' if tahmin[0] else 'Bulaşıcı Değil'}")
