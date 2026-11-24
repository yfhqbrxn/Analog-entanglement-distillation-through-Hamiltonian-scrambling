from pathlib import Path
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def binary_entropy(x, eps=1e-12):
    """
    Binary entropy function h(x) = -x log2 x - (1 - x) log2(1 - x).
    Values are clipped away from 0 and 1 for numerical stability.
    """
    x_arr = numpy.clip(x, eps, 1 - eps)
    return -(x_arr * numpy.log2(x_arr) + (1 - x_arr) * numpy.log2(1 - x_arr))

def calc(p, n, m):
    b = (1-p/2)**n
    eb0 = (p/2) / (1-b)
    eb = eb0 / (1 + b / (1-b) * (((2-3*p/2)/(1-p/2))**m))

    epn = p/4/(1-p/2)*((1-3*p/4)**m)*((1-p/2)**(n-m)) + \
        1/2 * (2**(-m)) * (1-b)
    epd = ((1-3*p/4)**m)*((1-p/2)**(n-m)) + (2**(-m)) * (1-b)
    ep = epn / epd
    return eb, ep 


def coherent_information(p, n, m):
    """
    Convenience wrapper that returns 1 - h(eb) - h(ep) from calc(p, n, m).
    """
    eb, ep = calc(p, n, m)
    return 1.0 - binary_entropy(eb) - binary_entropy(ep)


def find_zero_coherent_information(n, m, tol=1e-12, max_iter=100):
    """
    Find p in [0, 2/3] such that coherent_information(p, n, m) = 0 via bisection.
    Assumes the function is monotonic for fixed n,m.
    """
    low, high = 0.001, 2.0 / 3.0
    f_low = coherent_information(low, n, m)
    f_high = coherent_information(high, n, m)

    if f_low == 0:
        return low / 2
    if f_high == 0:
        return high / 2
    if f_low * f_high > 0:
        raise ValueError(
            "Coherent information does not cross zero within [0, 2/3] for n={}, m={}".format(
                n, m
            )
        )

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = coherent_information(mid, n, m)
        if abs(f_mid) < tol or (high - low) / 2 < tol:
            return mid / 2

        if f_mid * f_low > 0:
            low, f_low = mid, f_mid
        else:
            high, f_high = mid, f_mid

    return (0.5 * (low + high)) / 2


def plot_zero_surface(n_values, m_values, output_path=None):
    """
    Plot the surface p(n, m) where coherent_information(p, n, m) = 0.
    Values with m >= n (invalid region) are left blank.
    """
    if output_path is None:
        output_path = Path(__file__).resolve().parents[1] / "outputs" / "grid.svg"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    n_values = numpy.asarray(n_values)
    m_values = numpy.asarray(m_values)
    n_grid, m_grid = numpy.meshgrid(n_values, m_values, indexing="ij")
    z_grid = numpy.full_like(n_grid, numpy.nan, dtype=float)

    for i, n in enumerate(n_values):
        for j, m in enumerate(m_values):
            if m >= n:
                continue
            try:
                z_grid[i, j] = find_zero_coherent_information(int(n), int(m))
            except ValueError:
                # Leave invalid entries as NaN so they vanish from the surface.
                continue

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        n_grid,
        m_grid,
        z_grid,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.95,
    )
    ax.set_xlabel("n (total qubits)", fontsize=12)
    ax.set_ylabel("m (measured qubits)", fontsize=12)
    ax.set_zlabel("error rate", fontsize=12)
    ax.set_title("Tolerable Error Rate", fontsize=14)
    # Rotate so the (0, 0) corner appears on the left of the saved figure.
    ax.view_init(elev=28, azim=300)
    fig.colorbar(surf, shrink=0.65, pad=0.08, label="p")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # Use the original example grid; the historical paper grid is unverified.
    n_vals = numpy.arange(20, 101, 5)
    m_vals = numpy.arange(2, 80, 4)
    plot_zero_surface(n_vals, m_vals)