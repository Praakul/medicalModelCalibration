import torch
import os
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt

class CalibrationMetrics:
    def __init__(self, n_bins=15):
        self.n_bins = n_bins

    def _compute_static_bin_boundaries(self, device):
            # Helper for static (equal-width) bins
            return torch.linspace(0, 1, self.n_bins + 1, device=device)

    def _compute_adaptive_bin_boundaries(self, confidences):
        # Helper for adaptive (equal-mass) bins
        if confidences.numel() == 0:
            return torch.linspace(0, 1, self.n_bins + 1, device=confidences.device)

        # --- FIX: Vectorized quantile computation ---
        bin_n = confidences.numel() // self.n_bins
        if bin_n == 0: # Handle small sample size
            return torch.linspace(0, 1, self.n_bins + 1, device=confidences.device)

        sorted_confidences, _ = torch.sort(confidences)

        # Get indices for bin boundaries
        indices = torch.arange(0, self.n_bins, device=confidences.device) * bin_n
        bin_boundaries = sorted_confidences[indices]

        # Add the final 1.0
        return torch.cat([bin_boundaries, torch.tensor([1.0], device=confidences.device)])        

    def _get_probabilities(self, outputs, labels, logits=True):
        probabilities = F.softmax(outputs, dim=1) if logits else outputs
        confidences, predictions = torch.max(probabilities, dim=1)
        accuracies = predictions.eq(labels).float()
        return probabilities, confidences, predictions, accuracies

    def expected_calibration_error(self, outputs, labels, logits=True):
        # This is the standard ECE with STATIC (equal-width) bins
        probabilities, confidences, predictions, accuracies = self._get_probabilities(outputs, labels, logits)

        # --- FIX: Use helper and pass device ---
        bin_boundaries = self._compute_static_bin_boundaries(device=outputs.device)
        bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]

        ece = torch.tensor(0.0, device=outputs.device)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                confidence_in_bin = confidences[in_bin].mean()
                ece += prop_in_bin * torch.abs(confidence_in_bin - accuracy_in_bin)
        return ece.item()

    def max_calibration_error(self, outputs, labels, logits=True):
        probabilities, confidences, predictions, accuracies = self._get_probabilities(outputs, labels, logits)
        bin_boundaries = self._compute_static_bin_boundaries(device=outputs.device)
        bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]
        bin_scores = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                confidence_in_bin = confidences[in_bin].mean()
                bin_scores.append(torch.abs(confidence_in_bin - accuracy_in_bin))
        return max(bin_scores).item() if bin_scores else 0.0

    def adaptive_calibration_error(self, outputs, labels, logits=True):
        # This is ECE with ADAPTIVE (equal-mass) bins
        probabilities, confidences, predictions, accuracies = self._get_probabilities(outputs, labels, logits)

        # --- FIX: Use adaptive bin helper ---
        bin_boundaries = self._compute_adaptive_bin_boundaries(confidences)
        bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]

        ace = torch.tensor(0.0, device=outputs.device)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                confidence_in_bin = confidences[in_bin].mean()
                # --- FIX: Use the weighted average ---
                ace += prop_in_bin * torch.abs(confidence_in_bin - accuracy_in_bin)
        return ace.item()

    def compute_auc(self, outputs, labels, logits=True):
        probabilities, confidences, predictions, accuracies = self._get_probabilities(outputs, labels, logits)

    # Convert labels to one-hot encoding
        labels_one_hot = F.one_hot(labels, num_classes=probabilities.shape[1]).float()

    # Check if every class has at least one positive sample
        valid_classes = labels_one_hot.sum(dim=0) > 0  # Boolean mask for valid classes

        if valid_classes.sum() < 2:  # If fewer than 2 classes are present in this batch
           print("Warning: Not enough class diversity in batch for AUC. Skipping computation.")
           return torch.tensor(0.0, device=outputs.device)  # Return zero instead of crashing

    # Compute AUC only for valid classes
        auc_score = roc_auc_score(
        labels_one_hot[:, valid_classes].cpu().numpy(),
        probabilities[:, valid_classes].cpu().numpy(),
        multi_class='ovr',
        average='weighted'
        )

        return torch.tensor(auc_score, device=outputs.device)



    def compute_f1(self, outputs, labels, logits=True):
        probabilities,confidences,predictions,accuracies = self._get_probabilities(outputs, labels, logits)
        return f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='weighted')

    
    def plot_reliability_diagram(self, outputs, labels, logits=True):
        probabilities, confidences, predictions, accuracies = self._get_probabilities(outputs, labels, logits)
        bin_boundaries = self._compute_static_bin_boundaries(device=outputs.device)
        bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]
        bin_accuracies = []
        bin_confidences = []
     
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin > 0:
               bin_accuracies.append(accuracies[in_bin].mean().cpu().item())
               bin_confidences.append(confidences[in_bin].mean().cpu().item())

        fig, ax = plt.subplots(figsize=(6, 6))  # Create figure and axis
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        ax.scatter(bin_confidences, bin_accuracies, color='red', label='Model Reliability')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title('Reliability Diagram')
        ax.legend()

        return fig  # Return the figure object



