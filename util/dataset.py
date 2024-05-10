import pandas as pd
from llama_index.core import Document


def load_policies(
    file: str = "data/clean/encoded_policy.csv",
    start_index: int = 0,
    limit: int = 100,
    airline: str = None,
    as_dataframe: bool = False,
):
    # load dataframe:
    df = pd.read_csv(file)[start_index:start_index + limit]
    df["Content"] = df.apply(lambda row: f"{row['Header 1']}\n{row['Header 2']}\n{row['Concat']}".replace("\n\n", "\n").replace("\n\n", "\n"), axis=1)
    if airline is not None:
        df = df[df["Airline"] == airline]
    print(f"Loaded policies: {df.shape}")
    if as_dataframe:
        return df
    documents = [
        Document(
            id_=f"policy_{idx}",
            text=row["Content"],
            metadata={
                "airline": row["Airline"],
                "policy": row["Header 1"],
                "topic": row["Header 2"],
            }
        )
        for idx, row in df.iterrows()
    ]
    return documents
