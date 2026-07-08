import math
import bisect
import logging

# ---- Threshold Classes ----

# Class: Adaptive Threshold Model 1
class AdaptiveThreshold:
    """
    - Initialize the model with parameters or default values
    - beta: cost of misclassification
    - eta: learning rate for updating weights
    - quantize_step: step size for quantization
    - p_intervals: intervals for confidence scores
    - weights: weights for each interval
    """
    def __init__(self, beta=0.5, eta=0.06, quantize_step=0.01):
        self.beta = beta
        self.eta = eta
        self.quantize_step = quantize_step
        self.p_intervals = [0.0, 0.75, 1.0]
        self.weights = [1.0, 1.0]

    def _quantize(self, score):
        return round(score / self.quantize_step) * self.quantize_step

    def update_thresholds(self, confidence_score, correct_classification):
        confidence_score = self._quantize(confidence_score)
        confidence_score = min(max(confidence_score, 0.0), 1.0)

        idx = bisect.bisect_right(self.p_intervals, confidence_score) - 1

        if confidence_score not in self.p_intervals:
            self.p_intervals.insert(idx + 1, confidence_score)
            self.weights.insert(idx + 1, self.weights[idx])

        for i in range(len(self.weights)):
            if self.p_intervals[i+1] <= confidence_score:
                cost = self.beta if correct_classification == 0 else 0.0
            else:
                cost = self.beta if correct_classification == 1 else 0.0
            self.weights[i] *= math.exp(-self.eta * cost)

        total_weight = sum(
            (self.p_intervals[i+1] - self.p_intervals[i]) * self.weights[i]
            for i in range(len(self.weights))
        )
        self.weights = [w / total_weight for w in self.weights]

    def get_threshold(self):
        cumulative_weight = 0.0
        total_weight = sum(
            (self.p_intervals[i+1] - self.p_intervals[i]) * self.weights[i]
            for i in range(len(self.weights))
        )

        for i in range(len(self.weights)):
            cumulative_weight += (self.p_intervals[i+1] - self.p_intervals[i]) * self.weights[i]
            if cumulative_weight >= 0.5 * total_weight:
                adaptive_threshold = self.p_intervals[i+1]
                break
        else:
            adaptive_threshold = 1.0

        return adaptive_threshold
    
# Initialize Adaptive Threshold Model
"""
- beta: cost of misclassification
- eta: learning rate for updating weights
- quantize_step: step size for quantization
"""
adaptive_threshold_model = AdaptiveThreshold(beta=0.5, eta=0.06, quantize_step=0.01)


def reset_adaptive_threshold_model():
    global adaptive_threshold_model
    adaptive_threshold_model = AdaptiveThreshold(beta=0.5, eta=0.06, quantize_step=0.01)


def get_adaptive_threshold():
    return adaptive_threshold_model.get_threshold()

# ---- Main Function ----

# Function: Offloading Decision Maker
def make_offloading_decision(filename, confidence, decision_method, fixed_threshold):

    if decision_method == "never_offload":
        logging.info(f"Sample: {filename}, Offload decision: False (Never offload)")
        return False
    
    elif decision_method == "always_offload":
        logging.info(f"Sample: {filename}, Offload decision: True (Always offload)")
        return True
    
    elif decision_method == "fixed_threshold":
        decision = confidence < fixed_threshold
        logging.info(f"Sample: {filename}, Offload decision: {decision} (Confidence: {confidence:.2f} < Fixed threshold: {fixed_threshold:.2f})")
        return decision

    elif decision_method == "adaptive_threshold":
        adaptive_threshold = adaptive_threshold_model.get_threshold()
        decision = confidence < adaptive_threshold
        logging.info(f"Sample: {filename}, Offload decision: {decision} (Confidence: {confidence:.2f} < Adaptive threshold: {adaptive_threshold:.2f})")
        return decision

    else:
        return False
