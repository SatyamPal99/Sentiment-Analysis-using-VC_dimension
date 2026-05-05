def predict_texts(model, vectorizer, texts):
    """Predict a list of texts (or a single text) and return predictions as ints.

    `texts` can be a string or an iterable of strings.
    """
    single = False
    if isinstance(texts, str):
        texts = [texts]
        single = True

    X = vectorizer.transform(texts)
    preds = model.predict(X)

    return preds[0] if single else preds
