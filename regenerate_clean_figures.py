import pickle
import pandas as pd
from src import evaluation, config

with open('models/pipeline_outputs.pkl', 'rb') as f:
    data = pickle.load(f)

models_dict = data['models']
prep = data['prep_data']

X_test_dict = prep['X_test_dict']
y_test = prep['y_test']
label_encoder = prep['label_encoder']

models_probs = {}
for name, model in models_dict.items():
    if hasattr(model, 'predict_proba'):
        models_probs[name] = model.predict_proba(X_test_dict[name])
    else:
        preds = model.predict(X_test_dict[name])
        n_classes = len(label_encoder.classes_)
        probs = np.zeros((len(y_test), n_classes))
        for i, p in enumerate(preds):
            probs[i, p] = 1.0
        models_probs[name] = probs

print("Regenerating clean ROC, PR, and Calibration figures...")
evaluation.plot_calibration_curves(models_probs, y_test, label_encoder)
evaluation.plot_roc_curves(models_probs, y_test, label_encoder)
evaluation.plot_pr_curves(models_probs, y_test, label_encoder)
print("Figures regenerated successfully!")
