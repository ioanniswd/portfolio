import matplotlib.pyplot as plt
import re

def round_to_leftmost_digits(x, digits=2):
    num_of_digits = len(str(int(x)))
    return round(x, -num_of_digits + digits)

def pareto(viz_df, component_column, cumulative_column,
           annotate_cumulative=True, bold_key='bold'):
    """
    Create a Pareto chart from a DataFrame.

    Parameters
    ----------
    viz_df : pd.DataFrame
        DataFrame containing the data to be visualized.
    component_column : str
        Name of the column containing the components.
    cumulative_column : str
        Name of the column containing the cumulative values.
    annotate_cumulative : bool, optional, default True
        Whether to annotate the cumulative values on the chart.
    bold_key : str, optional, default 'bold'
        Font weight considered bold for the font.

    Returns
    -------
    None

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from visualization_helpers import pareto
    >>> np.random.seed(0)
    >>> viz_df = pd.DataFrame(
    ...     np.random.randint(1, 100, 10),
    ...     columns=['non_cumulative']
    ... )
    >>> viz_df['cumulative'] = viz_df['non_cumulative'].cumsum()
    >>> viz_df['component'] = [f'Component {x}' for x in range(1, 11)]
    >>> pareto(viz_df, 'component', 'cumulative') # plots the Pareto chart
    """
    viz_df = viz_df.copy()

    fig, ax = plt.subplots(figsize=(16, 6), dpi=110)

    GREEN = '#0C8040'
    BLUE = '#1f77b4'

    viz_df['non_cumulative'] = viz_df[cumulative_column] - viz_df[cumulative_column].shift(1).fillna(0)

    num_of_bars = len(viz_df[component_column])
    ax.set_xlim(-0.5, num_of_bars - 0.5)
    ax.set_xticks(range(num_of_bars))
    ax.set_xlabel(component_column, fontsize=16, color=BLUE, fontweight=bold_key)
    ax.bar(
            viz_df[component_column],
            viz_df['non_cumulative'],
            color=BLUE
            )

    # reduce size of xtick labels
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(13)
        tick.label1.set_color(BLUE)

    ax.set_ylabel(f'{cumulative_column}', fontsize=16, color=GREEN, fontweight=bold_key)
    ax.set_yticks(range(0, 101, 10))
    ytick_labels = [f'{x}%' for x in range(0, 101, 10)]
    ax.set_yticklabels(ytick_labels)

    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(13)
        tick.label1.set_color(GREEN)

    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)

    ax.scatter(viz_df[component_column], viz_df[cumulative_column], color=GREEN, s=12)

    # set title
    ax.set_title(f"Pareto Chart of {cumulative_column}", fontsize=20, color=GREEN,
                 fontweight=bold_key)

    ax.plot(viz_df[component_column], viz_df[cumulative_column], color=GREEN)

    if annotate_cumulative:
        for i, (_, row) in enumerate(viz_df.iterrows()):
            if i == 0:
                continue

            ax.annotate(
                    f"{round((row[cumulative_column]))}%",
                    xy=(
                        row[component_column],
                        row[cumulative_column] + 1
                        ),
                    ha='center',
                    va='bottom',
                    fontsize=11,
                    color=GREEN
                    )

    for i, row in viz_df.iterrows():
        ax.annotate(
                f"{round_to_leftmost_digits(row['non_cumulative'])}%",
                xy=(
                    row[component_column],
                    row['non_cumulative']
                    ),
                ha='center',
                va='bottom',
                fontsize=11,
                color=BLUE
                )


    ax.set_xticks(range(num_of_bars))

    # 45 degree rotation of xtick labels if needed
    longest_x_label_length = viz_df[component_column].str.len().max()

    if longest_x_label_length >= 10:
        plt.xticks(rotation=90)
    elif longest_x_label_length >= 5:
        plt.xticks(rotation=45)

    plt.show()

