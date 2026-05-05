import matplotlib.pyplot as plt
import seaborn as sns


def plot_vc_bound(emp_error, h, N, model_type):
    """Return a matplotlib Figure showing empirical error and VC bound."""
    fig, ax = plt.subplots(figsize=(6, 4))
    bound = None
    # the caller computes the bound; here we just plot emp_error and bound if provided
    # but to preserve original behavior, caller will pass computed bound through emp_error and h
    ax.bar(["Empirical Error", "VC Bound"], [emp_error, h], color=["#4CAF50", "#FF9800"])
    ax.set_ylabel("Error")
    ax.set_title(f"VC-Dimension Generalization Bound ({model_type})")
    return fig


def make_vc_figure(emp_error, bound, model_type):
    """Create the VC bar plot (empirical error and bound) and return Figure."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Empirical Error", "VC Bound"], [emp_error, bound], color=["#4CAF50", "#FF9800"])
    ax.set_ylabel("Error")
    ax.set_title(f"VC-Dimension Generalization Bound ({model_type})")
    return fig


def plot_train_test_errors(train_error, test_error, model_type):
    """Plot training and test error as side-by-side bars and return Figure."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Train Error", "Test Error"], [train_error, test_error], color=["#2196F3", "#FF5722"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Error")
    ax.set_title(f"Train vs Test Error ({model_type})")
    return fig


def plot_confusion_matrix(cm):
    """Return a matplotlib Figure for the confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])
    return fig


def plot_metrics_comparison(metrics_dict):
    """Plot grouped bar chart comparing metrics across datasets.

    metrics_dict: {dataset_name: {metric_name: value, ...}, ...}
    Returns a matplotlib Figure.
    """
    import numpy as _np

    datasets = list(metrics_dict.keys())
    metric_names = [k for k in next(iter(metrics_dict.values())).keys()]

    # build data matrix (len(metric_names) x len(datasets))
    data = _np.array([[metrics_dict[ds][m] for ds in datasets] for m in metric_names])

    x = _np.arange(len(metric_names))
    width = 0.8 / max(1, len(datasets))

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, ds in enumerate(datasets):
        ax.bar(x + i * width, data[:, i], width, label=ds)

    ax.set_xticks(x + width * (len(datasets) - 1) / 2)
    ax.set_xticklabels(metric_names)
    # scale Y axis between min and max values (with small padding) for better visibility
    all_vals = data.flatten()
    min_val = float(all_vals.min())
    max_val = float(all_vals.max())
    if max_val == min_val:
        pad = 0.02
    else:
        pad = max(0.02, (max_val - min_val) * 0.1)
    bottom = max(0.0, min_val - pad)
    top = min(1.0, max_val + pad)
    ax.set_ylim(bottom, top)
    ax.set_ylabel("Score")
    ax.set_title("Model Metrics Comparison Across Datasets")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_vc_accuracy_curves(results_dict, title="VC-dimension vs Accuracy"):
    """Plot VC-dimension (x= max_features) vs training and testing accuracy for multiple datasets.

    results_dict: {dataset_name: {"xs": [...], "train": [...], "test": [...]}, ...}
    """
    import numpy as _np

    fig, ax = plt.subplots(figsize=(8, 5))

    for ds_name, vals in results_dict.items():
        xs = vals["xs"]
        train = vals["train"]
        test = vals["test"]
        ax.plot(xs, train, marker="o", label=f"{ds_name} Train")
        ax.plot(xs, test, marker="s", linestyle="--", label=f"{ds_name} Test")

    ax.set_xlabel("TF-IDF max_features (proxy for VC-dimension)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    # determine global min/max across all series to improve visual sensitivity
    all_acc = []
    for vals in results_dict.values():
        all_acc.extend(vals.get("train", []))
        all_acc.extend(vals.get("test", []))
    if len(all_acc) == 0:
        ax.set_ylim(0.0, 1.0)
    else:
        min_acc = min(all_acc)
        max_acc = max(all_acc)
        if max_acc == min_acc:
            pad = 0.02
        else:
            pad = max(0.02, (max_acc - min_acc) * 0.1)
        bottom = max(0.0, min_acc - pad)
        top = min(1.0, max_acc + pad)
        ax.set_ylim(bottom, top)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
