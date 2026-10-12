
import joblib

model = joblib.load("phishing_model.joblib")

test_urls = [
    "https://www.google.com",
    "https://www.microsoft.com",
    "https://www.wikipedia.org",
    "http://example.com",
    "http://www.dghjdgf.com/paypal.co.uk/cycgi-bin/webscrcmd=_home-customer&nav=1/loading.php",
]

for url in test_urls:
    prediction = model.predict([url])[0]
    probabilities = model.predict_proba([url])[0]

    phishing_index = list(model.classes_).index(1)
    score = probabilities[phishing_index] * 100

    label = "PHISHING" if prediction == 1 else "LEGITIMATE"

    print("-" * 60)
    print("URL:", url)
    print("Prediction:", label)
    print(f"Phishing model score: {score:.2f}%")
