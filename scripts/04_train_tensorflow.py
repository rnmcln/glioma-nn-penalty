"""Step 04 - fit and characterise the final TensorFlow MLP on the full TCGA cohort.

The fair internal comparison happens in step 03 (shared folds). This step trains
the deployable TF model on all labelled TCGA patients for each feature set,
records the architecture/hyperparameters table, and saves the model object and
training history. It does NOT tune the network outside cross-validation.
"""
import os, sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from src import config, pipeline
from src.endpoints import make_binary_endpoint
from src.models_tensorflow import TFMLPClassifier
from src.utils import get_logger, set_global_seed

log = get_logger()


def main():
    set_global_seed()
    endpoint = os.environ.get("GNN_ENDPOINT", config.PRIMARY_ENDPOINT)
    horizon = config.ENDPOINTS[endpoint]
    df = pipeline.load_cohort("tcga")
    labelled, info = make_binary_endpoint(df, horizon)

    from src.models_tensorflow import ARCHITECTURES
    arch_desc = {
        "compact": "Dense32 - Dropout0.2 - Dense16 - Dropout0.2 - sigmoid",
        "medium": "Dense128 - BN - Dropout0.3 - Dense64 - Dropout0.3 - Dense32 - Dropout0.3 - sigmoid (L2 1e-4)",
        "regularised": "Dense64 - Dropout0.4 - Dense32 - Dropout0.4 - sigmoid (L2 1e-3)",
    }
    hp_rows = []
    for fs in ["A_clinical", "B_clinical_molecular", "C_expression"]:
        feats = pipeline.resolve_features(fs, df)
        for arch in ["compact", "medium", "regularised"]:
            clf = TFMLPClassifier(features=feats, arch=arch, seed=config.SEED)
            clf.fit(labelled, labelled["y"].values)
            clf.model_.save(config.MODEL_OBJECTS / f"tf_{arch}_{fs}_{endpoint}.keras")
            hist = clf.model_.history.history
            hp_rows.append({
                "variant": f"tf_{arch}", "architecture": arch_desc[arch], "feature_set": fs,
                "n_input_features_after_encoding": clf.n_input_,
                "l2": ARCHITECTURES[arch]["l2"], "optimizer": "Adam", "learning_rate": clf.lr,
                "loss": "binary_crossentropy", "batch_size": clf.batch_size,
                "max_epochs": clf.max_epochs, "early_stopping_patience": clf.patience,
                "epochs_trained": len(hist["loss"]), "class_weighted": True, "seed": clf.seed,
            })
            log.info("TF %s fit | set=%s epochs=%d", arch, fs, len(hist["loss"]))

    pd.DataFrame(hp_rows).to_csv(config.TABLES / "supplement_model_details.csv", index=False)
    log.info("Wrote supplement_model_details.csv and saved Keras models.")


if __name__ == "__main__":
    main()
