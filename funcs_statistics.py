"""Wrappers around statistical tests"""

# Builtin
from typing import TypedDict, Optional, cast
from textwrap import dedent

# 3rd party
import numpy as np
import pandas as pd
import scipy.stats
from scikit_posthocs import posthoc_tukey


class AnovaTukeyResults(TypedDict):
    fstat: float
    anova_p: float
    tukey: pd.DataFrame


def anova_tukey(
    df: pd.DataFrame, val_col: str, group_col: str, title: Optional[str] = None
) -> AnovaTukeyResults:
    ## ANOVA one-way
    groupdata = [
        cast(pd.Series, df.loc[groups, val_col]).values for groups in df.index.unique()
    ]

    fstat, pval = scipy.stats.f_oneway(*groupdata, axis=0)
    posthoc = posthoc_tukey(df.reset_index(), val_col=val_col, group_col=group_col)

    print(
        dedent(
            f"""
            ANOVA TUKEY STATS {title or ''}
            {fstat = :.2f}, {pval = :.3g}
            {posthoc}
            """
        )
    )

    return {"fstat": fstat, "anova_p": pval, "tukey": posthoc}


def t_test_ind(df1: pd.DataFrame, df2: pd.DataFrame, val_col: str):
    index_name = "_".join(df1.index.names)
    t_test_results = []

    for index_value in np.unique(df1.index):
        t_stats, p = scipy.stats.ttest_ind(
            df1.loc[index_value, val_col],
            df2.loc[index_value, val_col],
        )
        print(index_value, p)
        t_test_results.append(
            {
                index_name: index_value,
                "t_stat": t_stats,
                "p": p,
            }
        )
    return pd.DataFrame(t_test_results).set_index(index_name)
