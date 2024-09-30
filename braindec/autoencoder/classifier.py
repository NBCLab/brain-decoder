import numpy as np
import torch
import xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from braindec.autoencoder.model import MRI3dAutoencoder


class LabeledMRIDataset(Dataset):
    def __init__(self, num_samples=1000, image_shape=(64, 64, 64), num_classes=3):
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.num_classes = num_classes
        # Generate random 3D images and labels for demonstration purposes
        self.data = np.random.rand(num_samples, 1, *image_shape).astype(np.float32)
        self.labels = np.random.randint(0, num_classes, num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


def extract_latent_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Extracting features"):
            inputs = inputs.to(device)
            encoded, _ = model.encoder(inputs)
            # Flatten the encoded features
            batch_features = encoded.view(encoded.size(0), -1).cpu().numpy()
            features.append(batch_features)
            labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)


def hyperparameter_tuning(X_train, y_train):
    # Define the hyperparameter search space
    param_dist = {
        "n_estimators": randint(100, 1000),
        "max_depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.3),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "gamma": uniform(0, 0.5),
    }

    # Create an XGBoost classifier
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")

    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        n_iter=100,  # Number of parameter settings sampled
        cv=5,  # 5-fold cross-validation
        verbose=2,
        random_state=42,
        n_jobs=-1,  # Use all available cores
    )

    # Perform random search
    random_search.fit(X_train, y_train)

    # Print the best parameters and score
    print("Best parameters found: ", random_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(random_search.best_score_))

    return random_search.best_estimator_


def main():
    # Hyperparameters
    num_classes = 3
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained autoencoder
    autoencoder = MRI3dAutoencoder().to(device)
    autoencoder.load_state_dict(torch.load("mri_3d_autoencoder.pth"))

    # Create dataset and data loader
    dataset = LabeledMRIDataset(
        num_samples=1000, image_shape=(64, 64, 64), num_classes=num_classes
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Extract features using the pre-trained encoder
    features, labels = extract_latent_features(autoencoder, dataloader, device)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # Perform hyperparameter tuning
    print("Starting hyperparameter tuning...")
    best_xgb_clf = hyperparameter_tuning(X_train, y_train)

    # Train the model with the best parameters
    print("Training the model with the best parameters...")
    best_xgb_clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = best_xgb_clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained XGBoost model
    best_xgb_clf.save_model("mri_3d_xgboost_classifier_tuned.json")


if __name__ == "__main__":
    main()
