"""Predicts the output of the model on the test data."""

import numpy as np
import torch
import xgboost as xgb

from braindec.autoencoder.model import MRI3dAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
autoencoder = MRI3dAutoencoder()
autoencoder.load_state_dict(torch.load("mri_3d_autoencoder.pth"))

# Load the trained XGBoost classifier
best_xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    n_jobs=-1,
    random_state=42,
)
best_xgb_clf.load_model("best_xgb_clf.model")


# Example of using the model for prediction
def predict(model, autoencoder, image, device):
    autoencoder.eval()
    with torch.no_grad():
        input_tensor = torch.from_numpy(image).unsqueeze(0).to(device)  # Add batch dimension
        encoded, _ = autoencoder.encoder(input_tensor)
        features = encoded.view(1, -1).cpu().numpy()
        return model.predict(features)[0]


# Generate a sample image for prediction
sample_image = np.random.rand(1, 64, 64, 64).astype(np.float32)
predicted_label = predict(best_xgb_clf, autoencoder, sample_image, device)
print(f"\nPredicted label for sample image: {predicted_label}")
