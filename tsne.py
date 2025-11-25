import numpy as np

class TSNE:
    def __init__(self, perplexity, eta, tol: float=1e-2, seed: int | None=None) -> None:
        self.perplexity = perplexity
        self.eta = eta
        self.tol= tol
        self.rng = np.random.default_rng(seed)

    def compute_d(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the squared pairwise L2 distance
        """
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        np.fill_diagonal(D, 0)

        return np.maximum(D, 0)

    def compute_p(self, X: np.ndarray) -> None:
        """
        Compute the symmetrized P
        """
        H_target = np.log(self.perplexity)
        D = self.compute_d(X)
        self.P = np.zeros((self.n, self.n))
        P_i = np.zeros(self.n)

        max_tries = 50 # for the binary search for sigma
        log_lim = 1e-4

        for i in range(self.n):
            sigma2_min = -np.inf
            sigma2_max = np.inf
            sigma2 = 1.0

            for step in range(max_tries):
                d_i = D[i, :]
                P_i = np.exp(-d_i / sigma2)
                P_i[i] = 0

                # normalize P and compute entropy
                sum_P_i = np.sum(P_i)
                if sum_P_i == 0: sum_P_i = 1e-10
                P_i /= np.sum(P_i)

                P_i_n0 = P_i[P_i > log_lim]
                H_current = -np.sum(P_i_n0 * np.log(P_i_n0))

                H_diff = H_current - H_target

                if np.abs(H_diff) < self.tol:
                    # print(f"Break (step = {step})")
                    break

                if H_diff > 0:
                    # entropy too big => lower sigma2
                    sigma2_max = sigma2

                    if sigma2_min == -np.inf:
                        sigma2 /= 2
                    else:
                        sigma2 = (sigma2_min + sigma2) / 2
                else:
                    # entropy too low => bigger sigma2
                    sigma2_min = sigma2

                    if sigma2_max == float('inf'):
                        sigma2 *= 2
                    else:
                        sigma2 = (sigma2 + sigma2_max) / 2

            # copy over final values
            self.P[i, :] = P_i[:]

        # symmetrize P
        self.P = (self.P + self.P.T) / (2 * self.n)
        self.P = np.maximum(self.P, 1e-10)

    def init_random_state(self) -> None:
        """
        Generate a new random state
        """
        self.Y = self.rng.normal(0, 1e-4, (self.n, 2))

    def compute_q(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Q_ij for low-dimensional points
        """
        D_Y = self.compute_d(self.Y)
        inv_D = 1.0 / (1.0 + D_Y)
        np.fill_diagonal(inv_D, 0)

        return inv_D / np.sum(inv_D), inv_D


    def compute_grad(self) -> None:
        """
        Compute the gradient
        """
        # grad = 4 * sum((p_ij - q_ij) * (y_i - y_j) * inv_D_ij)
        # grad_i = 4 * sum_j (scalar_ij * (y_i - y_j))
        #        = 4 * (y_i * sum_j scalar_ij - sum_j (scalar_ij * y_j))
        Q, inv_D = self.compute_q()

        coeff = 1
        if self.current_step <= 100:
            coeff = 4

        PQ_diff = (coeff * self.P - Q) * inv_D
        sum_PQ = np.sum(PQ_diff, axis=1).reshape(-1, 1)

        self.grad = 4 * (sum_PQ * self.Y - np.dot(PQ_diff, self.Y))

    def step(self) -> None:
        """
        Go through one step of gradiant descent
        """
        self.current_step += 1

        momentum = 0.5
        if self.current_step > 100:
            momentum = 0.8

        self.compute_grad()

        self.Y_step = -self.eta * self.grad + momentum * self.Y_step
        self.Y += self.Y_step

        self.Y -= np.mean(self.Y, axis=0)

    def init(self, X: np.ndarray) -> None:
        self.n = X.shape[0]
        self.dim = 2
        self.compute_p(X)
        self.init_random_state()
        self.Y_step = np.zeros((self.n, 2))
        self.current_step = 0

    def fit(self, X: np.ndarray, steps: int) -> np.ndarray:
        """
        Fit datas onto a 2D map using pre-preconfigured t-SNE
        """
        self.init(X)

        for i in range(steps):
            self.step()

            if i % 20 == 0:
                print(f"Step {i}/{steps}")

        return self.Y

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    tsne = TSNE(perplexity=10, eta=50)
    steps = 200

    X1 = np.random.normal(0, 1, (30, 3))
    X2 = np.random.normal(20, 1, (30, 3))
    # X3 = np.random.normal(40, 1, (30, 3))
    X = np.vstack([X1, X2])
    colors = np.concatenate([np.zeros(30), np.ones(30)])

    fig = plt.figure(figsize=(16, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, cmap='viridis', marker='o')
    ax1.set_title('3D View of Clusters')
    # ax1.set_xlabel('X Axis')
    # ax1.set_ylabel('Y Axis')
    # ax1.set_zlabel('Z Axis')

    tsne.init(X)

    ax2 = fig.add_subplot(122)
    scatter2 = ax2.scatter([], [], c=[], cmap='viridis', marker='o')
    ax2.set_title(f't-SNE Evolution : Step 0/{steps}')
    # ax2.set_xlabel('X Axis')
    # ax2.set_ylabel('Y Axis')
    # ax2.grid(True)
    # ax2.set_aspect('equal', adjustable='box')

    skip = 1
    def update(frame):
        for _ in range(skip):
            tsne.step()

        Y = tsne.Y
        scatter2.set_offsets(Y)
        scatter2.set_array(colors)

        xmin, xmax = np.min(Y[:, 0]), np.max(Y[:, 0])
        ymin, ymax = np.min(Y[:, 1]), np.max(Y[:, 1])
        x_range, y_range = xmax - xmin, ymax - ymin
        buffer = 0.05 # 5 percent buffer
        new_xlim = (xmin - buffer * x_range, xmax + buffer * x_range)
        new_ylim = (ymin - buffer * y_range, ymax + buffer * y_range)

        ax2.set_xlim(new_xlim)
        ax2.set_ylim(new_ylim)
        ax2.set_title(f"t-SNE Evolution : Step {tsne.current_step}/{steps}")

        return scatter2, 

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=steps // skip,
        interval=50,
        blit=False,
    )

    plt.tight_layout()
    plt.show()
