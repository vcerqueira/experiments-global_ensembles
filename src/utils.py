import pandas as pd


def _escape_latex(text):
    return str(text).replace('_', r'\_')


def _higher_is_better(idx, higher_is_better):
    parts = idx if isinstance(idx, tuple) else (idx,)
    if higher_is_better is None:
        return any(str(part).startswith('Top-') for part in parts)
    return idx in higher_is_better or any(part in higher_is_better for part in parts)


def _format_index(index):
    if isinstance(index, pd.MultiIndex):
        return [
            r'\_'.join(_escape_latex(part) for part in levels if str(part) != '')
            for levels in index
        ]
    return [_escape_latex(x) for x in index]


def _rank_marks(values, maximize, mark_second):
    ranked = values.dropna().sort_values(ascending=not maximize).unique()
    best = ranked[0] if len(ranked) else None
    second = ranked[1] if mark_second and len(ranked) > 1 else None
    return best, second


def _format_cell(value, maximize, round_to_n, best, second):
    if pd.isna(value):
        return ''
    cell = f'{int(value)}' if maximize else f'{value:.{round_to_n}f}'
    if value == best:
        return rf'\textbf{{{cell}}}'
    if second is not None and value == second:
        return rf'\underline{{{cell}}}'
    return cell


def to_latex_tab(
    df,
    round_to_n=3,
    rotate_cols=False,
    caption='CAPTION',
    label='LABEL',
    higher_is_better=None,
    mark_second=True,
    axis=1,
):
    """Render a numeric table as LaTeX, bolding the best and underlining the second-best.

    By default the best value is taken along each row (``axis=1``). Pass
    ``axis=0`` to mark the best value in each column instead.

    Rows whose last index level starts with ``Top-`` (or labels passed in
    ``higher_is_better``) treat larger values as better and print as integers.
    """
    table = df.copy()
    numeric = table.select_dtypes(include='number')
    rows = []
    if axis == 0:
        col_state = {}
        for col in numeric.columns:
            values = numeric[col].round(round_to_n)
            best, second = _rank_marks(values, maximize=False, mark_second=mark_second)
            col_state[col] = (values, best, second)
        for idx in numeric.index:
            formatted = {}
            for col in numeric.columns:
                values, best, second = col_state[col]
                formatted[col] = _format_cell(
                    values.loc[idx], False, round_to_n, best, second
                )
            rows.append(formatted)
    else:
        for idx, row in numeric.iterrows():
            maximize = _higher_is_better(idx, higher_is_better)
            values = row.round(0 if maximize else round_to_n)
            best, second = _rank_marks(values, maximize, mark_second)
            rows.append(
                {
                    col: _format_cell(value, maximize, round_to_n, best, second)
                    for col, value in values.items()
                }
            )
    annotated = pd.DataFrame(rows, index=numeric.index, columns=numeric.columns)

    annotated.index = _format_index(annotated.index)
    columns = [_escape_latex(c) for c in annotated.columns]
    if rotate_cols:
        columns = [rf'\rotatebox{{60}}{{{c}}}' for c in columns]
    annotated.columns = columns

    n_cols = annotated.shape[1]
    return annotated.to_latex(
        escape=False,
        caption=caption,
        label=label,
        column_format='l' + 'r' * n_cols,
        index=True,
    )
