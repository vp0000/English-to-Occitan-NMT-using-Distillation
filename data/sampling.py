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
        train_size: int,
        distill_size: int,
        train_earlystop_size: int,
        distill_earlystop_size: int,
        random_seed: int =  42
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Set the random seed for reproducibility
    random_state = random_seed

    # Create a new column for the length of the text
    for col in df.columns:
        if df[col].dtype == 'object':
            df[f'{col}_text_length'] = df[col].apply(lambda x: len(str(x)))
    # Calculate the average text length for each row
    text_length_cols = [col for col in df.columns if col.endswith('_text_length')]
    df['avg_text_length'] = df[text_length_cols].mean(axis='columns')

    # Create stratified bins based on the average text length
    df['length_bin'] = pd.qcut(df['avg_text_length'], q=10, labels=False)
    # Sample the training set using stratified sampling
    train_df = df.groupby('length_bin', group_keys=False).apply(lambda x: x.sample(n=int(train_size / 10), random_state=random_state))
    remaining_df = df.drop(train_df.index)

    # Sample the distillation set from the remaining data using stratified sampling
    distill_df = remaining_df.groupby('length_bin', group_keys=False).apply(lambda x: x.sample(n=int(distill_size / 10), random_state=random_state))
    remaining_df = remaining_df.drop(distill_df.index)

    # Sample the early stopping set for training from the remaining data using stratified sampling
    train_earlystop_df = remaining_df.groupby('length_bin', group_keys=False).apply(lambda x: x.sample(n=int(train_earlystop_size / 10), random_state=random_state))
    remaining_df = remaining_df.drop(train_earlystop_df.index)

    # Sample the early stopping set for distillation from the remaining data using stratified sampling
    distill_earlystop_df = remaining_df.groupby('length_bin', group_keys=False).apply(lambda x: x.sample(n=int(distill_earlystop_size / 10), random_state=random_state))
    # Drop the auxiliary columns
    for col in df.columns:
        if col.endswith('_text_length') or col == 'avg_text_length' or col == 'length_bin':
            df.drop(columns=col, inplace=True)
    return train_df, distill_df, train_earlystop_df, distill_earlystop_df

def perform_sampling(
        df: pd.DataFrame,
        train_size: int,
        distill_size: int,
        train_earlystop_size: int,
        distill_earlystop_size: int,
        sampling_strategy: str = "random",
        random_seed: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
            train_size=train_size,
            distill_size=distill_size,
            train_earlystop_size=train_earlystop_size,
            distill_earlystop_size=distill_earlystop_size,
            random_seed=random_seed
        )
    else:
        raise ValueError("Invalid sampling strategy. Please choose 'random' or 'length_stratified'.") 