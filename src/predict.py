# Prediction logic

import pickle
from preprocess import preprocess_text

# loading the saved vectorizer and models from artifacts
with open("artifacts/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("artifacts/lr_model.pkl", "rb") as f:
    lr_model, lr_threshold = pickle.load(f)

with open("artifacts/nb_model.pkl", "rb") as f:
    nb_model, nb_threshold = pickle.load(f)


def predict(text, model="lr"):
    """
    Predict whether news is Fake or True
    using Logistic Regression or Naive Bayes.
    """

    clean = preprocess_text(text)
    vec = vectorizer.transform([clean])

    if model == "lr":
        prob = lr_model.predict_proba(vec)[0, 1]
        return "True" if prob >= lr_threshold else "Fake"

    elif model == "nb":
        prob = nb_model.predict_proba(vec)[0, 1]
        return "True" if prob >= nb_threshold else "Fake"

    else:
        raise ValueError("model must be 'lr' or 'nb'")


if __name__ == "__main__":
    print("Fake News Detection")
    print("Choose model: lr (Logistic Regression) or nb (Naive Bayes)")
    model_choice = input("Enter model (lr/nb): ").strip().lower()

    if model_choice not in ["lr", "nb"]:
        print("Invalid model choice. Defaulting to Logistic Regression.")
        model_choice = "lr"

    print("\nEnter the news text:")
    user_text = input("> ").strip()

    if not user_text:
        print("No text entered. Exiting.")
    else:
        prediction = predict(user_text, model=model_choice)
        print("\nPrediction:", prediction)