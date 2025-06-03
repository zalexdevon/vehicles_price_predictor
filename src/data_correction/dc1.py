import pandas as pd
import numpy as np
from Mylib import myfuncs
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


special_chars = ["#", "*", "x"]


def is_not_valid(value):
    if pd.isnull(value):
        return False

    for char in special_chars:
        if char in value:
            return True

    return False


def replace_not_valid_value_by_nan(col):
    index_not_valid = col.index[col.apply(is_not_valid)]
    col[index_not_valid] = np.nan
    return col


def replace_value_l0_by_nan(col):
    value_l0_index = col.index[col < 0]
    col[value_l0_index] = np.nan
    return col


class BeforeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        df = X

        # Xóa các cột không liên quan
        df = df.drop(columns=["time", "station", "AMB_TEMP", "time_datetime"])

        # Xóa các cột có tỉ lệ missing lớn
        ## Trong các cột (ngoại trừ cột mục tiêu), thay thế tất cả các giá trị mà có chứa 1 trong 3 kí tự đặc biệt sau: ['#', '*', 'x'] thành np.nan
        special_chars = ["#", "*", "x"]
        target_col = "temperature_cat"
        feature_cols = list(set(df.columns.tolist()) - set([target_col]))
        df[feature_cols] = df[feature_cols].apply(replace_not_valid_value_by_nan)
        df["CH4"].unique()

        ## Các cột có tỉ lệ missing lớn > 10% là: UVB, PH_RAIN, RAIN_COND, NMHC, CH4, THC
        df = df.drop(columns=["UVB", "PH_RAIN", "RAIN_COND", "NMHC", "CH4", "THC"])

        # Đổi tên cột
        rename_dict = {
            "CO": "CO_num",
            "NO": "NO_num",
            "NO2": "NO2_num",
            "NOx": "NOx_num",
            "O3": "O3_num",
            "PM10": "PM10_num",
            "PM2.5": "PM2_5_num",
            "RAINFALL": "RAINFALL_num",
            "RH": "RH_num",
            "SO2": "SO2_num",
            "WD_HR": "WD_HR_num",
            "WIND_DIREC": "WIND_DIREC_num",
            "WIND_SPEED": "WIND_SPEED_num",
            "WS_HR": "WS_HR_num",
            "temperature_cat": "temperature_cat_target",
        }

        df = df.rename(columns=rename_dict)

        # Sắp xếp các cột theo đúng thứ tự
        (
            numeric_cols,
            numericCat_cols,
            cat_cols,
            binary_cols,
            nominal_cols,
            ordinal_cols,
            target_col,
        ) = myfuncs.get_different_types_cols_from_df_4(df)

        df = df[
            numeric_cols
            + numericCat_cols
            + binary_cols
            + nominal_cols
            + ordinal_cols
            + [target_col]
        ]

        # Kiểm tra kiểu dữ liệu các cột
        feature_cols = list(set(df.columns.tolist()) - set([target_col]))

        ## Thay thế các giá trị 'NR' trong cột RAINFALL_num bằng 0
        col_name = "RAINFALL_num"
        replace_list = [(["NR"], 0)]
        df[col_name] = myfuncs.replace_in_series_33(df[col_name], replace_list)

        ## Thay thế các giá trị 'NR' trong cột PM2_5_num bằng 0
        col_name = "PM2_5_num"
        replace_list = [(["NR"], 0)]
        df[col_name] = myfuncs.replace_in_series_33(df[col_name], replace_list)

        ## Chuyển các cột sai dữ liệu về kiểu dữ liệu float
        df[feature_cols] = df[feature_cols].astype("float32")

        # Kiểm tra nội dung các cột numeric
        ## Thay thế các giá trị âm trong các cột này thành np.nan
        col_names = ["NO_num", "NO2_num", "O3_num", "SO2_num"]
        df[col_names] = df[col_names].apply(replace_value_l0_by_nan)

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
