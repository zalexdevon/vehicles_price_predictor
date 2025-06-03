from sklearn.base import BaseEstimator, TransformerMixin


class TransformerOnTrainAndTest(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        df = X

        # Chuyển kiểu dữ liệu cột year_nom từ int sang string
        col_name = "year_nom"
        df[col_name] = df[col_name].astype("string")

        # Ứng với từng cột trên (ngoại trừ cột numcat), thay thế các giá trị không nằm trong các label xuất hiện nhiều nhất thành 'other'
        most_frequent_values_dict = {
            "year_nom": ["2012", "2013", "2014"],
            "make_nom": ["Ford", "Chevrolet", "Nissan"],
            "model_nom": ["Altima"],
            "trim_nom": ["Base", "SE"],
            "body_nom": ["Sedan", "SUV"],
            "transmission_nom": ["automatic"],
            "state_nom": ["fl", "ca", "pa"],
            "color_nom": ["black", "white", "silver", "gray"],
            "interior_nom": ["black", "gray"],
            "seller_nom": [
                "nissan-infiniti lt",
                "ford motor credit company llc",
                "the hertz corporation",
            ],
        }

        for col_name, most_frequent_values in most_frequent_values_dict.items():
            df[col_name] = df[col_name].str.strip()  # Loại bỏ khoảng trắng thừa
            index_non_most_frequent = df[col_name].index[
                ~df[col_name].isin(most_frequent_values)
            ]
            df.loc[index_non_most_frequent, col_name] = "other"

        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class TransformerOnTrain(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        df = X

        self.is_fitted_ = True
        return df

    def transform(self, X, y=None):

        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
