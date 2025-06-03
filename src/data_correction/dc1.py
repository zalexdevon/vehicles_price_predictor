import pandas as pd
import numpy as np
from Mylib import myfuncs
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline




class BeforeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        df = X

        # Xóa các cột không liên quan
        df = df.drop(
            columns=[
                "vin", 'sellingprice', 'saledate'
            ]
        )

        # Đổi tên cột
        rename_dict = {
            "year": "year_nom",
            "make": "make_nom",
            "model": "model_nom",
            "trim": "trim_nom",
            "body": "body_nom",
            "transmission": "transmission_nom",
            "state": "state_nom",
            "condition": "condition_numcat",
            "odometer": "odometer_num",
            "color": "color_nom",
            "interior": "interior_nom",
            "seller": "seller_nom",
            "mmr": "mmr_num",
            "sellingprice_cat": "sellingprice_cat_target",


        }


        df = df.rename(columns=rename_dict)

        # Sắp xếp các cột theo đúng thứ tự
        numeric_cols, numericCat_cols, cat_cols, binary_cols, nominal_cols, ordinal_cols, target_col = myfuncs.get_different_types_cols_from_df_4(df)


        df = df[
            numeric_cols
            + numericCat_cols
            + binary_cols
            + nominal_cols
            + ordinal_cols
            + [target_col]
        ]



        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.cols


class MissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        df = X

        numeric_cols, numericCat_cols, cat_cols, _, _, _, target_col = (
            myfuncs.get_different_types_cols_from_df_4(df)
        )

        self.handler = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="mean"), numeric_cols),
                (
                    "numCat",
                    SimpleImputer(strategy="most_frequent"),
                    numericCat_cols,
                ),
                ("cat", SimpleImputer(strategy="most_frequent"), cat_cols),
                ("target", SimpleImputer(strategy="most_frequent"), [target_col]),
            ]
        )
        self.handler.fit(df)
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        df = X

        numeric_cols, numericCat_cols, cat_cols, _, _, _, target_col = (
            myfuncs.get_different_types_cols_from_df_4(df)
        )

        df = self.handler.transform(df)
        self.cols = numeric_cols + numericCat_cols + cat_cols + [target_col]
        df = pd.DataFrame(df, columns=self.cols)

        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.cols


class AfterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        df = X

        self.cols = df.columns.tolist()

        numeric_cols, numericCat_cols, cat_cols, _, _, _, target_col = (
            myfuncs.get_different_types_cols_from_df_4(df)
        )

        # Chuyển đổi về đúng kiểu dữ liệu
        df[numeric_cols] = df[numeric_cols].astype("float32")
        df[numericCat_cols] = df[numericCat_cols].astype("float32")
        df[cat_cols] = df[cat_cols].astype("category")
        df[target_col] = df[target_col].astype("category")

        # Loại bỏ duplicates
        df = df.drop_duplicates().reset_index(drop=True)

        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.cols
