from tsne import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def generate_linked_rings(n_per_ring=300, radius=1.0, noise=0.01):
    """
    Generates two interlocking rings in 3D
    """
    theta1 = np.linspace(0, 2 * np.pi, n_per_ring)
    # x = cos, y = sin, z = small noise
    r1_x = radius * np.cos(theta1) - (radius / 2) # Shift left slightly
    r1_y = radius * np.sin(theta1)
    r1_z = np.random.normal(0, noise, n_per_ring)

    ring1 = np.column_stack([r1_x, r1_y, r1_z])

    theta2 = np.linspace(0, 2 * np.pi, n_per_ring)
    # x = cos, z = sin, y = small noise
    # Shift x right so they interlock like a chain
    r2_x = radius * np.cos(theta2) + (radius / 2)
    r2_z = radius * np.sin(theta2)
    r2_y = np.random.normal(0, noise, n_per_ring)

    ring2 = np.column_stack([r2_x, r2_y, r2_z])

    # Combine
    X = np.vstack([ring1, ring2])

    # Labels for coloring (0 for ring1, 1 for ring2)
    labels = np.concatenate([np.zeros(n_per_ring), np.ones(n_per_ring)])

    return X, labels

if __name__ == "__main__":
    n_per_ring = 300
    perplexity = 50
    learning_rate = 50
    steps_per_frame = 5
    total_frames = 120

    print("Generating Linked Rings...")
    X, labels = generate_linked_rings(n_per_ring, radius=1.0)

    print("Initializing t-SNE...")
    tsne = TSNE(perplexity=perplexity, eta=learning_rate, seed=42)
    tsne.init(X)

    fig = plt.figure(figsize=(15, 7))

    ax1 = fig.add_subplot(121, projection='3d')
    # Use a discrete colormap: Ring 1 = Blue, Ring 2 = Red
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='coolwarm', s=20, edgecolors='k', linewidth=0.2)
    ax1.set_title("Input: 3D Interlocked Rings")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_zlim(-2, 2)

    ax2 = fig.add_subplot(122)
    scatter2 = ax2.scatter([], [], c=[], cmap='coolwarm', s=20, edgecolors='k', linewidth=0.2)
    ax2.set_title("Output: t-SNE Unwinding")
    ax2.grid(True, alpha=0.3)

    def update(frame):
        for _ in range(steps_per_frame):
            tsne.step()

        Y = tsne.Y
        scatter2.set_offsets(Y)
        scatter2.set_array(labels)

        xmin, xmax = np.min(Y[:, 0]), np.max(Y[:, 0])
        ymin, ymax = np.min(Y[:, 1]), np.max(Y[:, 1])
        pad_x = (xmax - xmin) * 0.1 if xmax != xmin else 1.0
        pad_y = (ymax - ymin) * 0.1 if ymax != ymin else 1.0

        ax2.set_xlim(xmin - pad_x, xmax + pad_x)
        ax2.set_ylim(ymin - pad_y, ymax + pad_y)

        phase = "Early Exaggeration" if tsne.current_step < 100 else "Fine Tuning"
        ax2.set_title(f"Step {tsne.current_step} | Phase: {phase}")

        return scatter2,

    print("Starting Animation...")
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=30, blit=False)
    plt.show()
