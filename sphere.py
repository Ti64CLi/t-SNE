from tsne import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def generate_sphere_points(n_points=100, radius=1.0):
    """
        Generates points uniformly on a sphere surface
    """
    points = np.random.randn(n_points, 3)
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]

    return points * radius

if __name__ == "__main__":
    n_points = 500
    radius = 1.0
    perplexity = 50
    learning_rate = 50
    steps_per_frame = 5
    total_frames = 100

    print("Generating Sphere Data...")
    X = generate_sphere_points(n_points, radius=radius)
    colors = X[:, 2]

    print("Initializing t-SNE...")
    tsne = TSNE(perplexity=perplexity, eta=learning_rate, seed=42)
    tsne.init(X)

    fig = plt.figure(figsize=(15, 7))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, cmap='plasma', s=20, edgecolors='k', linewidth=0.2)
    ax1.set_title("Input: 3D Sphere")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(122)
    scatter = ax2.scatter([], [], c=[], cmap='plasma', s=30, edgecolors='k', linewidth=0.2)
    ax2.set_title('t-SNE Optimization')
    ax2.grid(True, alpha=0.3)

    def update(frame):
        for _ in range(steps_per_frame):
            tsne.step()

        Y = tsne.Y

        scatter.set_offsets(Y)
        scatter.set_array(colors)

        xmin, xmax = np.min(Y[:, 0]), np.max(Y[:, 0])
        ymin, ymax = np.min(Y[:, 1]), np.max(Y[:, 1])
        x_range, y_range = xmax - xmin, ymax - ymin

        ax2.set_xlim(xmin - 0.1 * x_range, xmax + 0.1 * x_range)
        ax2.set_ylim(ymin - 0.1 * y_range, ymax + 0.1 * y_range)

        phase = "Early Exaggeration" if tsne.current_step <= 100 else "Fine Tuning"
        ax2.set_title(f"Step {tsne.current_step} | Phase: {phase}")

        return scatter,

    print("Starting Animation")
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=30,
        blit=False
    )

    plt.tight_layout()
    plt.show()
