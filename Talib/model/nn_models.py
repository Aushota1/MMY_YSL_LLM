"""
Опциональный LSTM-классификатор для ранга. Требует tensorflow.
При отсутствии библиотеки get_lstm_classifier возвращает None.
"""

from typing import Any, Callable, Optional

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False


class LSTMWrapper:
    """Обёртка для предсказания через загруженную Keras-модель (совместимость с sklearn-интерфейсом)."""

    def __init__(self, model_path: str, window_len: int, n_features: int, n_quantiles: int):
        self.model_path = model_path
        self.window_len = window_len
        self.n_features = n_features
        self.n_quantiles = n_quantiles
        self._model = None

    def _get_model(self):
        if self._model is None and HAS_KERAS:
            self._model = keras.models.load_model(self.model_path)
        return self._model

    def predict(self, X: Any) -> Any:
        import numpy as np
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_feat = self.window_len * self.n_features
        if X.shape[1] != n_feat:
            X = X[:, :n_feat] if X.shape[1] > n_feat else X
        X_3d = X.reshape(X.shape[0], self.window_len, self.n_features)
        m = self._get_model()
        if m is None:
            return np.zeros(X.shape[0], dtype=int)
        proba = m.predict(X_3d, verbose=0)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: Any) -> Any:
        import numpy as np
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_feat = self.window_len * self.n_features
        if X.shape[1] != n_feat:
            X = X[:, :n_feat] if X.shape[1] > n_feat else X
        X_3d = X.reshape(X.shape[0], self.window_len, self.n_features)
        m = self._get_model()
        if m is None:
            return np.zeros((X.shape[0], self.n_quantiles))
        return m.predict(X_3d, verbose=0)


def build_and_fit_lstm(
    X_train: Any,
    y_train: Any,
    window_len: int,
    n_features: int,
    n_quantiles: int,
    save_path: str,
    epochs: int = 50,
    batch_size: int = 32,
    units: int = 64,
    use_bidirectional: bool = False,
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
) -> Optional[LSTMWrapper]:
    """
    Строит и обучает LSTM (два слоя: units и units//2), сохраняет в save_path (.keras).
    """
    if not HAS_KERAS:
        return None
    import numpy as np
    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.int32)
    n_feat = window_len * n_features
    if X.shape[1] != n_feat:
        X = X[:, :n_feat]
    X_3d = X.reshape(X.shape[0], window_len, n_features)
    units2 = max(16, units // 2)
    if use_bidirectional:
        layer1 = layers.Bidirectional(layers.LSTM(units, return_sequences=True))
        layer2 = layers.LSTM(units2)
    else:
        layer1 = layers.LSTM(units, return_sequences=True)
        layer2 = layers.LSTM(units2)
    model = keras.Sequential([
        layers.Input(shape=(window_len, n_features)),
        layer1,
        layers.Dropout(0.2),
        layer2,
        layers.Dropout(0.2),
        layers.Dense(n_quantiles, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = []
    if progress_callback is not None:
        def on_epoch_end(epoch, logs=None):
            logs = logs or {}
            progress_callback(
                epoch + 1,
                epochs,
                float(logs.get("loss", 0)),
                float(logs.get("accuracy", 0)),
            )
        callbacks.append(keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end))
    model.fit(
        X_3d, y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )
    model.save(save_path)
    return LSTMWrapper(save_path, window_len, n_features, n_quantiles)


def get_lstm_classifier(
    X_train: Any,
    y_train: Any,
    window_len: int,
    n_quantiles: int,
    save_path_keras: str,
    epochs: int = 50,
    batch_size: int = 32,
    units: int = 64,
    use_bidirectional: bool = False,
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
) -> Optional[Any]:
    """
    Возвращает обученный LSTMWrapper или None, если Keras недоступен.
    """
    if not HAS_KERAS:
        return None
    n = X_train.shape[1]
    n_features = n // window_len
    if n_features * window_len != n:
        return None
    return build_and_fit_lstm(
        X_train, y_train,
        window_len=window_len,
        n_features=n_features,
        n_quantiles=n_quantiles,
        save_path=save_path_keras,
        epochs=epochs,
        batch_size=batch_size,
        units=units,
        use_bidirectional=use_bidirectional,
        progress_callback=progress_callback,
    )
