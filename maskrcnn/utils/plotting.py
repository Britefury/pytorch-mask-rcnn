import numpy as np
from matplotlib.patches import Rectangle
from skimage.color import hsv2rgb
from skimage.filters import roberts


############################################################
#  Plotting Utility Functions
############################################################

def plot_boxes(ax, boxes, alpha=1.0, colours=None):
    if colours is None:
        colours = ['red'] * len(boxes)
    for (y1, x1, y2, x2), col in zip(boxes, colours):
        rect = Rectangle((x1, y1), x2-x1, y2-y1, facecolor=None, edgecolor=col, fill=False, alpha=alpha)
        ax.add_patch(rect)


def plot_stratified_boxes(ax, boxes, alpha=1.0, colour='red'):
    for group in boxes:
        group_colour = group.get('colour', colour)
        group_alpha = group.get('alpha', alpha)
        for (y1, x1, y2, x2) in group['boxes']:
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor=None, edgecolor=group_colour, fill=False, alpha=group_alpha)
            ax.add_patch(rect)


def labels_to_rgb(labels, label_to_rgb_table=None, mark_edges=False):
    if label_to_rgb_table is None:
        hue = np.random.uniform(low=0.0, high=1.0, size=(labels.max(), 1, 1))
        sat = np.random.uniform(low=0.5, high=1.0, size=(labels.max(), 1, 1))
        val = np.ones((labels.max(), 1, 1))
        hsv = np.concatenate([hue, sat, val], axis=2)
        rgb = hsv2rgb(hsv)[:, 0, :]
        label_to_rgb_table = np.append([[0.0, 0.0, 0.0]], rgb, axis=0)

    labels_rgb = label_to_rgb_table[labels]

    if mark_edges:
        edges = roberts(labels) > 0
        labels_rgb[edges, :] = 0

    return labels_rgb


def visualise_labels_rgb(X, labels, labels_rgb, mark_edges=False):
    alpha = (labels > 0)[:, :, None] * 0.3
    vis = (X * (1.0 - alpha)) + labels_rgb * alpha

    if mark_edges:
        edges = roberts(labels) > 0
        mask = (1.0 - edges * 0.7)

        return vis * mask[:, :, None]
    else:
        return vis


def visualise_labels(X, labels, label_to_rgb_table=None, mark_edges=False):
    labels_rgb = labels_to_rgb(labels, label_to_rgb_table=label_to_rgb_table, mark_edges=mark_edges)
    return visualise_labels_rgb(X, labels, labels_rgb, mark_edges=mark_edges)