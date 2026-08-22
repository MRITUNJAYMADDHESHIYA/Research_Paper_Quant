# data_loader.py

import pandas as pd


class DataLoader:

    def __init__(self, path):
        self.path = path

    def load(self):

        df = pd.read_csv(self.path)

        df.columns = [
            col.strip()
            for col in df.columns
        ]

        print("\nOriginal columns:")
        print(df.columns.tolist())

        return df

    def prepare(self, df):

        # ==========================================
        # DATE
        # ==========================================

        df["Date"] = pd.to_datetime(
            df["Date"],
            format="%d-%m-%Y",
            errors="coerce"
        )

        # ==========================================
        # RENAME COLUMNS
        # ==========================================

        df = df.rename(
            columns={
                "Price": "Close",
                "Vol.": "Volume"
            }
        )

        # ==========================================
        # PRICE COLUMNS
        # ==========================================

        price_columns = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        for col in price_columns:

            df[col] = (
                df[col]
                .astype(str)
                .str.replace(
                    ",",
                    "",
                    regex=False
                )
            )

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        # ==========================================
        # VOLUME
        # ==========================================

        def convert_volume(value):

            if pd.isna(value):
                return None

            value = str(value).strip()

            if value.endswith("K"):

                return (
                    float(
                        value[:-1]
                    ) * 1_000
                )

            elif value.endswith("M"):

                return (
                    float(
                        value[:-1]
                    ) * 1_000_000
                )

            elif value.endswith("B"):

                return (
                    float(
                        value[:-1]
                    ) * 1_000_000_000
                )

            return float(value)

        df["Volume"] = (
            df["Volume"]
            .apply(convert_volume)
        )

        # ==========================================
        # CHANGE %
        # ==========================================

        if "Change %" in df.columns:

            df["Change %"] = (
                df["Change %"]
                .astype(str)
                .str.replace(
                    "%",
                    "",
                    regex=False
                )
            )

            df["Change %"] = pd.to_numeric(
                df["Change %"],
                errors="coerce"
            )

        # ==========================================
        # SORT DATE
        # ==========================================

        # Your data is newest -> oldest.
        # Convert it to oldest -> newest.

        df = df.sort_values(
            "Date"
        )

        # ==========================================
        # REMOVE DUPLICATES
        # ==========================================

        df = df.drop_duplicates(
            subset=["Date"]
        )

        # ==========================================
        # CALCULATE RETURN
        # ==========================================

        df["Return"] = (
            df["Close"].pct_change()
        )

        # ==========================================
        # DISPLAY MISSING VALUES
        # ==========================================

        print(
            "\nMissing values before cleaning:"
        )

        print(
            df[
                [
                    "Date",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "Return"
                ]
            ].isna().sum()
        )

        # ==========================================
        # REMOVE INVALID ROWS
        # ==========================================

        df = df.dropna(
            subset=[
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Return"
            ]
        )

        df = df.reset_index(
            drop=True
        )

        # ==========================================
        # FINAL INFORMATION
        # ==========================================

        print(
            "\nRows after cleaning:",
            len(df)
        )

        print(
            "\nCleaned data:"
        )

        print(
            df.head()
        )

        print(
            "\nData types:"
        )

        print(
            df.dtypes
        )

        return df