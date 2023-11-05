import matplotlib.pyplot as plt


def plot_metrics():
    """
    Plot the metrics for the baseline and t5 models.
    """
    baseline = [0.7739, 0.7307, 0.7219, 0.3861, 0.5060]
    t5 = [0.7993, 0.6640, 0.7617, 0.4180, 0.4627]
    metrics = ['ACC', 'SIM', 'FL', 'J', 'BLEU']

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(metrics, baseline, label='baseline')
    ax.plot(metrics, t5, label='t5')

    ax.set_title('Metrics')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.legend()

    plt.show()
