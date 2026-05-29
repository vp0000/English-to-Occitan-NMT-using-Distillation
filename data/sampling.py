import pandas as pd

def random_sampling(
        df: pd.DataFrame,
        train_size: int,
        distill_size: int,
        train_earlystop_size: int,
        distill_earlystop_size: int,
        random_seed: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # Set the random seed for reproducibility
    random_state = random_seed

    # Sample the training set
    train_df = df.sample(n=train_size, random_state=random_state)
    remaining_df = df.drop(train_df.index)

    # Sample the distillation set from the remaining data
    distill_df = remaining_df.sample(n=distill_size, random_state=random_state)
    remaining_df = remaining_df.drop(distill_df.index)

    # Sample the early stopping set for training from the remaining data
    train_earlystop_df = remaining_df.sample(n=train_earlystop_size, random_state=random_state)
    remaining_df = remaining_df.drop(train_earlystop_df.index)

    # Sample the early stopping set for distillation from the remaining data
    distill_earlystop_df = remaining_df.sample(n=distill_earlystop_size, random_state=random_state)

    return train_df, distill_df, train_earlystop_df, distill_earlystop_df

def length_stratified_sampling(
        df: pd.DataFrame,
        text_cols: list,
        train_size: int,
        distill_size: int,
        train_earlystop_size: int,
        distill_earlystop_size: int,
        random_seed: int = 42,
        n_bins: int = 10
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    # text_cols = work.select_dtypes(include='object').columns.tolist()
    work['_avg_len'] = work[text_cols].apply(
        lambda r: sum(len(str(v)) for v in r) / len(r), axis=1
    )
    work['_bin'] = pd.qcut(work['_avg_len'], q=n_bins,
                           labels=False, duplicates='drop')

    def sample_n(source, n, seed):
        sizes = source.groupby('_bin').size()
        per_bin = (sizes / sizes.sum() * n).round().astype(int)
        per_bin = per_bin.combine(sizes, min)
        return pd.DataFrame(source.groupby('_bin', group_keys=False).apply(
            lambda g: g.sample(n=int(per_bin[g.name]), random_state=seed)
        ))

    train_df = sample_n(work, train_size, random_seed)
    remaining = work.drop(train_df.index)
    distill_df = sample_n(remaining, distill_size, random_seed + 1)
    remaining = remaining.drop(distill_df.index)
    train_es = sample_n(remaining, train_earlystop_size, random_seed + 2)
    remaining = remaining.drop(train_es.index)
    distill_es = sample_n(remaining, distill_earlystop_size, random_seed + 3)

    aux = ['_avg_len', '_bin']
    return (
    train_df.drop(columns=aux),
    distill_df.drop(columns=aux),
    train_es.drop(columns=aux),
    distill_es.drop(columns=aux)
)

def hybrid_sampling(df: pd.DataFrame, text_cols: list, score_col: str, train_size: int, distill_size: int,
                                train_earlystop_size: int, distill_earlystop_size: int,
                                random_seed: int = 42, n_bins: int = 10, top_k: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    if not isinstance(score_col, str): raise ValueError(f'Needed a string value, got {type(score_col)}')
    if score_col not in work.columns:
        raise ValueError('Score column not present. Please check the input file.')
    # text_cols = work.select_dtypes(include='object').columns.tolist()
    work['_avg_len'] = work[text_cols].apply(
        lambda r: sum(len(str(v)) for v in r) / len(r), axis=1
    )
    work['_bin'] = pd.qcut(work['_avg_len'], q=n_bins,
                           labels=False, duplicates='drop')

    def sample_hybrid(source, n, seed):
        sizes = source.groupby('_bin').size()
        per_bin = (sizes / sizes.sum() * n).round().astype(int)
        per_bin = per_bin.combine(sizes, min)
        sampled_dfs = []
        for bin_id in source['_bin'].unique():
            bin_size = int(per_bin[bin_id])
            if bin_size > 0:
                bin_data = source[source['_bin'] == bin_id]
                top_count = int(bin_size * top_k)
                random_count = bin_size - top_count
                top_samples = bin_data.nlargest(top_count, score_col) if top_count > 0 else pd.DataFrame()
                if random_count > 0:
                    remaining_in_bin = bin_data.drop(top_samples.index) if len(top_samples) > 0 else bin_data
                    random_samples = remaining_in_bin.sample(
                        n = min(random_count, len(remaining_in_bin)),
                        random_state = seed
                    )
                    sampled_dfs.append(pd.concat([top_samples, random_samples]))
                else:
                    sampled_dfs.append(top_samples)
        return pd.concat(sampled_dfs) if sampled_dfs else pd.DataFrame()

    train_df = sample_hybrid(work, train_size, random_seed)
    remaining = work.drop(train_df.index)
    distill_df = sample_hybrid(remaining, distill_size, random_seed + 1)
    remaining = remaining.drop(distill_df.index)
    train_es = sample_hybrid(remaining, train_earlystop_size, random_seed + 2)
    remaining = remaining.drop(train_es.index)
    distill_es = sample_hybrid(remaining, distill_earlystop_size, random_seed + 3)

    aux = ['_avg_len', '_bin']
    # aux.append(score_col)
    return (
    train_df.drop(columns=aux),
    distill_df.drop(columns=aux),
    train_es.drop(columns=aux),
    distill_es.drop(columns=aux)
)

def perform_sampling(
        df: pd.DataFrame,
        text_cols: list,
        score_col: str,
        train_size: int,
        distill_size: int,
        train_earlystop_size: int,
        distill_earlystop_size: int,
        sampling_strategy: str = "random",
        random_seed: int = 42,
        top_k: float = 0.7,
        score_flag: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not score_flag and sampling_strategy == "hybrid":
        raise ValueError("Invalid sampling strategy. Cleaned data with score column required.")
    if sampling_strategy == "random":
        return random_sampling(
            df=df,
            train_size=train_size,
            distill_size=distill_size,
            train_earlystop_size=train_earlystop_size,
            distill_earlystop_size=distill_earlystop_size,
            random_seed=random_seed
        )
    elif sampling_strategy == "length_stratified":
        return length_stratified_sampling(
            df=df,
            text_cols=text_cols,
            train_size=train_size,
            distill_size=distill_size,
            train_earlystop_size=train_earlystop_size,
            distill_earlystop_size=distill_earlystop_size,
            random_seed=random_seed
        )
    elif sampling_strategy == "hybrid":
        return hybrid_sampling(
            df=df,
            text_cols=text_cols,
            score_col=score_col,
            train_size=train_size,
            distill_size=distill_size,
            train_earlystop_size=train_earlystop_size,
            distill_earlystop_size=distill_earlystop_size,
            random_seed=random_seed,
            top_k=top_k
        )
    else:
        raise ValueError("Invalid sampling strategy. Please choose 'random' or 'length_stratified'.") 