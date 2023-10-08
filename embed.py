import argparse
import os

import pandas as pd
import requests

from closure import Closure

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tx",
        type=str,
        default="https://tx.ontoserver.csiro.au/fhir",
        metavar="[FHIR terminology server endpoint]",
    )
    parser.add_argument(
        "--vs",
        type=str,
        required=True,
        metavar="[FHIR value set URI]",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        metavar="[batch size]",
        dest="batch_size",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        metavar="[output file]",
    )
    args = parser.parse_args()

    closure = Closure(tx_url=args.tx)

    print(f"Generating transitive closure for value set: {args.vs}")
    offset = 0
    total = 0
    feature_names = []
    while offset <= total:
        response = requests.get(
            f"{args.tx}/ValueSet/$expand",
            params={"url": args.vs, "count": args.batch_size, "offset": offset},
        )
        response.raise_for_status()
        response_json = response.json()

        print(
            f"Updating closure ({len(response_json['expansion']['contains'])} items, offset {offset})"
        )
        codings = response_json["expansion"]["contains"]
        closure.update(codings)
        feature_names.extend(
            [f"<< {coding['code']}|{coding['display']}|" for coding in codings]
        )

        total = response_json["expansion"]["total"]
        offset += args.batch_size

    # Create a Pandas DataFrame from the closure.
    print("Creating dataframe from closure")
    closure_df = pd.DataFrame(closure.to_array(), columns=["source", "target"])

    # One-hot encode the target column.
    print("One-hot encoding target column...", end=" ")
    ohe_df = pd.get_dummies(closure_df["target"])
    print(ohe_df.shape)

    # Add the source column as the index, then group by the source and sum the one-hot encoded
    # target columns. This results in a multi-hot encoded vector.
    print("Grouping by source and summing...", end=" ")
    ohe_df.set_index(closure_df["source"], inplace=True)
    grouped = ohe_df.groupby("source").sum()
    result = grouped.set_axis(feature_names, axis=1)
    print(result.shape)
    print(f"Sample:\n{result.head()}")

    # Save the dict to a pickle file.
    print(f"Saving to file: {args.output}", end=" ")
    pd.to_pickle(result, args.output, compression={"method": "gzip"})
    print(f"({os.path.getsize(args.output)} bytes written)")
