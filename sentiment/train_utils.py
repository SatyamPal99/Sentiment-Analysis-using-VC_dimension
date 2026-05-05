import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


def get_model_filenames(model_type, max_features, C, model_dir="saved_models", dataset_name=None):
    os.makedirs(model_dir, exist_ok=True)
    ds_tag = f"_{str(dataset_name).replace(' ', '_')}" if dataset_name else ""
    model_name = f"{model_type.replace(' ', '_')}_VC{max_features}_C{C}{ds_tag}.joblib"
    vectorizer_name = f"vectorizer_{model_type.replace(' ', '_')}_VC{max_features}{ds_tag}.joblib"
    model_path = os.path.join(model_dir, model_name)
    vectorizer_path = os.path.join(model_dir, vectorizer_name)
    return model_path, vectorizer_path, model_name, vectorizer_name


def load_or_train_model(data, model_type, max_features, C, model_dir="saved_models", dataset_name=None):
    """Load cached model/vectorizer if present and compatible, otherwise train and save.

    Returns: (model, vectorizer, model_name, vectorizer_name, was_loaded)
    """
    X_text = data["text"]
    y = data["label"]

    model_path, vectorizer_path, model_name, vectorizer_name = get_model_filenames(
        model_type, max_features, C, model_dir, dataset_name=dataset_name
    )

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        try:
            vectorizer = joblib.load(vectorizer_path)
            model = joblib.load(model_path)
            # verify compatibility
            if len(vectorizer.get_feature_names_out()) == max_features:
                return model, vectorizer, model_name, vectorizer_name, True
            # incompatible, fallthrough to retrain
        except Exception:
            # fallthrough to retrain
            pass

    # Train new model
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X = vectorizer.fit_transform(X_text)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    if model_type == "Logistic Regression":
        model = LogisticRegression(C=C, solver="saga", max_iter=1000)
    else:
        model = LinearSVC(C=C, max_iter=2000)

    model.fit(X_train, y_train)

    # Save for reuse
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    return model, vectorizer, model_name, vectorizer_name, False
