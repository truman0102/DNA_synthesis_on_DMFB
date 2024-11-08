import numpy as np
from matplotlib import pyplot as plt

def _process_goal(goal):
    return [goal[0]+1, goal[1]+1, goal[3] - goal[1] + 1, goal[2] - goal[0] + 1]

def plot(img, goals, frontier_colors):
    n_cols, n_rows, _ = img.shape
    fig, ax = plt.subplots()
    ax.imshow(
        np.flipud(img),
        extent=[0, n_cols + 2, 0, n_rows + 2],
        interpolation="nearest",
    )
    ax.invert_yaxis()
    plt.xticks(
        range(n_cols+2),
        [""] * (n_cols+2),
    )
    plt.yticks(
        range(n_rows+2),
        [""] * (n_rows+2),
    )
    plt.grid(True, which='both', axis='both', linestyle='--',  alpha=0.5)
    frontiers = [_process_goal(goal) for goal in goals]
    for frontier, l_color in zip(frontiers, frontier_colors):
        plt.gca().add_patch(
            plt.Rectangle(
                (frontier[1], frontier[0]),
                frontier[2],
                frontier[3],
                fill=False,
                edgecolor=l_color[0],
                linewidth=.5,
                linestyle="--",
            )
        )
        continue
        if len(l_color) > 1:
            for i in range(1, len(l_color)):
                interval = (i // 2 + 1) / 5
                if i % 2 == 0:
                    plt.gca().add_patch(
                        plt.Rectangle(
                            (frontier[1] + interval, frontier[0] - interval),
                            frontier[2],
                            frontier[3],
                            fill=False,
                            edgecolor=l_color[i],
                            linewidth=1,
                            linestyle="--",
                        )
                    )
                else:
                    plt.gca().add_patch(
                        plt.Rectangle(
                            (frontier[1] - interval, frontier[0] + interval),
                            frontier[2],
                            frontier[3],
                            fill=False,
                            edgecolor=l_color[i],
                            linewidth=1,
                            linestyle="--",
                        )
                    )