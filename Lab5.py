import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold

def init_seeds(s=42):
    np.random.seed(s)
    tf.random.set_seed(s)

def prepare_dataset(x_data, y_data, target_classes, limit=None, total_cap=None, s=42):
    init_seeds(s)
    y_flat = y_data.flatten()
    
    mask = np.isin(y_flat, target_classes)
    x_sel = x_data[mask]
    y_sel = y_flat[mask]

    label_map = {val: idx for idx, val in enumerate(target_classes)}
    y_mapped = np.array([label_map[v] for v in y_sel], dtype=int)

    final_indices = []
    for i in range(len(target_classes)):
        cls_indices = np.where(y_mapped == i)[0]
        np.random.shuffle(cls_indices)
        if limit is not None:
            cls_indices = cls_indices[:limit]
        final_indices.append(cls_indices)

    all_idx = np.concatenate(final_indices) if len(final_indices) > 0 else np.array([], dtype=int)
    np.random.shuffle(all_idx)

    if total_cap is not None:
        all_idx = all_idx[:total_cap]

    return x_sel[all_idx], y_mapped[all_idx]

def standardize_images(train_x, test_x):
    train_x = train_x.astype(np.float32)
    test_x = test_x.astype(np.float32)

    mu = train_x.mean(axis=(0, 1, 2), keepdims=True)
    sigma = train_x.std(axis=(0, 1, 2), keepdims=True) + 1e-7

    return (train_x - mu) / sigma, (test_x - mu) / sigma

def create_network(filters, classes_out, dense_drop=0.5, conv_drop=0.0, l_rate=1e-3):
    net = keras.models.Sequential()
    
    net.add(keras.layers.Input(shape=(32, 32, 3)))
    
    net.add(keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="same"))
    net.add(keras.layers.BatchNormalization(axis=-1))
    net.add(keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="same"))
    net.add(keras.layers.BatchNormalization(axis=-1))
    net.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    net.add(keras.layers.Dropout(conv_drop))

    net.add(keras.layers.Conv2D(2 * filters, (3, 3), activation="relu", padding="same"))
    net.add(keras.layers.BatchNormalization(axis=-1))
    net.add(keras.layers.Conv2D(2 * filters, (3, 3), activation="relu", padding="same"))
    net.add(keras.layers.BatchNormalization(axis=-1))
    net.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    net.add(keras.layers.Dropout(conv_drop))

    net.add(keras.layers.Conv2D(4 * filters, (3, 3), activation="relu", padding="same"))
    net.add(keras.layers.BatchNormalization(axis=-1))
    net.add(keras.layers.Conv2D(4 * filters, (3, 3), activation="relu", padding="same"))
    net.add(keras.layers.BatchNormalization(axis=-1))
    net.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    net.add(keras.layers.Dropout(conv_drop))

    net.add(keras.layers.Flatten())
    net.add(keras.layers.Dense(512, activation="relu"))
    net.add(keras.layers.BatchNormalization())
    net.add(keras.layers.Dropout(dense_drop))
    net.add(keras.layers.Dense(classes_out, activation="softmax"))

    net.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=l_rate),
        metrics=["accuracy"],
    )
    return net

def get_augmenter(rot, w_sh, h_sh, zm):
    return ImageDataGenerator(
        rotation_range=rot,
        width_shift_range=w_sh,
        height_shift_range=h_sh,
        zoom_range=zm,
    )

def evaluate_cv(x, y, folds, params, flt, bs, eps, s=42):
    init_seeds(s)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=s)
    scores = []

    for t_idx, v_idx in splitter.split(x):
        xt, xv = x[t_idx], x[v_idx]
        yt, yv = y[t_idx], y[v_idx]

        model = create_network(filters=flt, classes_out=y.shape[1])
        gen = get_augmenter(**params)
        gen.fit(xt)

        steps_per_ep = max(1, int(np.ceil(len(xt) / bs)))
        h = model.fit(
            gen.flow(xt, yt, batch_size=bs),
            steps_per_epoch=steps_per_ep,
            epochs=eps,
            validation_data=(xv, yv),
            verbose=0,
        )

        acc = float(np.max(h.history.get("val_accuracy", [0.0])))
        scores.append(acc)
        keras.backend.clear_session()

    return float(np.mean(scores)), scores

def run_search(x, y, n_tries, n_folds, flt, bs, ep, start_seed=42):
    init_seeds(start_seed)
    best_res = None
    log_history = []

    for i in range(n_tries):
        curr_params = {
            "rot": int(np.random.choice([0, 5, 10, 15, 20])),
            "w_sh": float(np.random.choice([0.0, 0.05, 0.1, 0.15])),
            "h_sh": float(np.random.choice([0.0, 0.05, 0.1, 0.15])),
            "zm": float(np.random.choice([0.0, 0.05, 0.1, 0.15])),
        }

        avg_score, fold_scores = evaluate_cv(x, y, n_folds, curr_params, flt, bs, ep, start_seed + i + 1)
        
        log_history.append((avg_score, curr_params, fold_scores))
        
        if best_res is None or avg_score > best_res[0]:
            best_res = (avg_score, curr_params, fold_scores)

        scores_formatted = [round(v, 4) for v in fold_scores]
        print(f"Try {i+1} | Avg: {avg_score:.4f} | Pars: {curr_params} | Sc: {scores_formatted}")

    log_history.sort(key=lambda x: x[0], reverse=True)
    return best_res, log_history

if __name__ == "__main__":
    init_seeds(42)

    (x_tr_raw, y_tr_raw), (x_te_raw, y_te_raw) = cifar10.load_data()

    sel_classes = [0, 1, 2, 3]
    trn_limit = 2000
    tst_limit = 800

    x_train, y_train = prepare_dataset(x_tr_raw, y_tr_raw, sel_classes, total_cap=trn_limit, s=42)
    x_test, y_test = prepare_dataset(x_te_raw, y_te_raw, sel_classes, total_cap=tst_limit, s=43)

    x_train_n, x_test_n = standardize_images(x_train, x_test)

    n_c = len(sel_classes)
    y_train_enc = to_categorical(y_train, n_c)
    y_test_enc = to_categorical(y_test, n_c)

    print("Data shapes:")
    print("Train:", x_train_n.shape, y_train_enc.shape)
    print("Test:", x_test_n.shape)
    print()

    iters = 6
    k_folds = 3
    n_filters = 32
    batch_sz = 64
    epochs_search = 3

    print("Starting optimization...")
    winner, _ = run_search(x_train_n, y_train_enc, iters, k_folds, n_filters, batch_sz, epochs_search, start_seed=100)

    best_avg, best_cfg, _ = winner
    print("\nBest config found:")
    print(f"Val Acc: {best_avg:.4f}")
    print(f"Params: {best_cfg}")
    print()

    epochs_fin = 8
    
    indices = np.random.permutation(len(x_train_n))
    split_pt = int(0.8 * len(x_train_n))
    idx_t, idx_v = indices[:split_pt], indices[split_pt:]

    xt, yt = x_train_n[idx_t], y_train_enc[idx_t]
    xv, yv = x_train_n[idx_v], y_train_enc[idx_v]

    final_model = create_network(n_filters, n_c)
    aug_gen = get_augmenter(**best_cfg)
    aug_gen.fit(xt)

    steps = max(1, int(np.ceil(len(xt) / batch_sz)))
    
    print("Training final model...")
    final_model.fit(
        aug_gen.flow(xt, yt, batch_size=batch_sz),
        steps_per_epoch=steps,
        epochs=epochs_fin,
        validation_data=(xv, yv),
        verbose=1,
    )

    loss_val, acc_val = final_model.evaluate(x_test_n, y_test_enc, verbose=0)
    print(f"\nFinal Test Accuracy: {float(acc_val):.4f}")