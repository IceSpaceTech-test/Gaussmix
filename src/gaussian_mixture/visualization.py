"""Visualization helpers for Gaussian mixtures."""
from __future__ import annotations
import numpy as np


def _get_covariance_matrix(model, k):
    if model.covariance_type == 'full':
        return np.asarray(model.covariances_[k], dtype=float)
    if model.covariance_type == 'tied':
        return np.asarray(model.covariances_, dtype=float)
    if model.covariance_type == 'diag':
        return np.diag(np.asarray(model.covariances_[k], dtype=float))
    if model.covariance_type == 'spherical':
        d = model.means_.shape[1]
        return np.eye(d) * float(model.covariances_[k])
    raise ValueError('Unsupported covariance_type: %s' % model.covariance_type)


class GMMVisualizer:
    def __init__(self, model):
        self.model = model

    def _ensure_matplotlib(self):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Ellipse
        except ImportError as exc:
            raise ImportError('matplotlib is required for visualization.') from exc
        return plt, Ellipse

    def _prepare_ax(self, ax, figsize=(7, 6)):
        plt, _ = self._ensure_matplotlib()
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            return fig, ax
        return None, ax

    def _grid_limits(self, X=None):
        means = np.asarray(self.model.means_, dtype=float)
        if X is not None:
            x_min, x_max = X[:, 0].min(), X[:, 0].max()
            y_min, y_max = X[:, 1].min(), X[:, 1].max()
        else:
            x_min, x_max = means[:, 0].min() - 3, means[:, 0].max() + 3
            y_min, y_max = means[:, 1].min() - 3, means[:, 1].max() + 3
        dx = (x_max - x_min) * 0.15
        dy = (y_max - y_min) * 0.15
        return x_min - dx, x_max + dx, y_min - dy, y_max + dy

    def plot_scatter(self, X=None, ax=None, show: bool = True, labels=None, cmap='tab10', alpha=0.6):
        plt, _ = self._ensure_matplotlib()
        fig, ax = self._prepare_ax(ax)
        if X is not None:
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if X.ndim != 2:
                raise ValueError('GMMVisualizer.scatter supports only 1D or 2D data.')

        if self.model.means_.ndim != 2 or self.model.means_.shape[1] not in (1, 2):
            raise ValueError('GMMVisualizer requires 1D or 2D model means.')

        if X is not None and X.shape[1] == 1:
            ax.hist(X[:, 0], bins=30, color='gray', alpha=0.6)
            ax.set_xlabel('Feature 0')
            ax.set_ylabel('Count')
        elif X is not None:
            ax.scatter(X[:, 0], X[:, 1], s=20, c=labels, cmap=cmap, alpha=alpha, label='data')
            ax.set_xlabel('Feature 0')
            ax.set_ylabel('Feature 1')
        else:
            ax.set_xlabel('Feature 0')
            ax.set_ylabel('Feature 1')

        means = np.asarray(self.model.means_, dtype=float)
        if means.shape[1] == 2:
            ax.scatter(means[:, 0], means[:, 1], c='red', marker='x', s=100, label='means')

        ax.legend()
        if show:
            plt.show()
        return ax

    def plot_decision_regions(self, X=None, ax=None, grid_size: int = 200, cmap='viridis', alpha: float = 0.4, levels: int = 10, show: bool = True):
        plt, _ = self._ensure_matplotlib()
        fig, ax = self._prepare_ax(ax)

        if self.model.means_.shape[1] != 2:
            raise ValueError('Decision regions are supported only for 2D models.')

        if X is not None:
            X = np.asarray(X, dtype=float)
            if X.ndim != 2 or X.shape[1] != 2:
                raise ValueError('Decision region plotting requires 2D data.')
            ax.scatter(X[:, 0], X[:, 1], s=20, color='gray', alpha=0.5, label='data')

        x_min, x_max, y_min, y_max = self._grid_limits(X)
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        zz = np.exp(self.model.score_samples(grid)).reshape(xx.shape)
        contourf = ax.contourf(xx, yy, zz, levels=levels, cmap=cmap, alpha=alpha)
        ax.contour(xx, yy, zz, levels=levels, colors='k', linewidths=0.5, alpha=0.6)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Feature 0')
        ax.set_ylabel('Feature 1')
        ax.set_title('GMM decision regions')
        if show:
            plt.show()
        return ax

    def plot_log_likelihood(self, history=None, ax=None, show: bool = True):
        plt, _ = self._ensure_matplotlib()
        fig, ax = self._prepare_ax(ax)
        if history is None:
            history = getattr(self.model, 'lower_bound_history_', None)
        if history is None:
            raise ValueError('No log-likelihood history is available for plotting.')
        ax.plot(np.arange(1, len(history) + 1), history, marker='o')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Log-likelihood')
        ax.set_title('EM log-likelihood trajectory')
        if show:
            plt.show()
        return ax

    def plot_component_density_panel(self, ax=None, ncols: int = 2, show: bool = True):
        plt, _ = self._ensure_matplotlib()
        n_components = int(self.model.n_components)
        ncols = max(1, int(ncols))
        nrows = int(np.ceil(n_components / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
        for k in range(n_components):
            row = k // ncols
            col = k % ncols
            current_ax = axes[row][col]
            mean = np.asarray(self.model.means_[k], dtype=float)
            cov = _get_covariance_matrix(self.model, k)
            if self.model.means_.shape[1] == 1:
                x = np.linspace(mean[0] - 4 * np.sqrt(cov[0, 0]), mean[0] + 4 * np.sqrt(cov[0, 0]), 200)
                pdf = np.exp(-0.5 * ((x - mean[0]) ** 2) / cov[0, 0]) / np.sqrt(2 * np.pi * cov[0, 0])
                current_ax.plot(x, pdf, label=f'Component {k}')
                current_ax.set_xlabel('Feature 0')
                current_ax.set_ylabel('Density')
            else:
                x_min, x_max, y_min, y_max = self._grid_limits(None)
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
                grid = np.column_stack([xx.ravel(), yy.ravel()])
                log_prob = self.model._estimate_log_prob(grid, self.model.means_, self.model.covariances_)
                zz = np.exp(log_prob[:, k]).reshape(xx.shape)
                current_ax.contour(xx, yy, zz, levels=8, cmap='viridis', alpha=0.8)
                current_ax.scatter(mean[0], mean[1], c='red', marker='x')
                current_ax.set_xlabel('Feature 0')
                current_ax.set_ylabel('Feature 1')
            current_ax.set_title(f'Component {k} density')
            current_ax.legend()
        for k in range(n_components, nrows * ncols):
            fig.delaxes(axes[k // ncols][k % ncols])
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    @staticmethod
    def save_figure(fig, path: str, dpi: int = 100):
        fig.savefig(path, dpi=dpi, bbox_inches='tight')


def plot_gmm(model, X=None, ax=None, show: bool = True, contour: bool = False, n_levels: int = 10):
    visualizer = GMMVisualizer(model)
    if contour:
        return visualizer.plot_decision_regions(X=X, ax=ax, levels=n_levels, show=show)
    return visualizer.plot_scatter(X=X, ax=ax, show=show)
