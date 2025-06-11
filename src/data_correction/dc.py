from Mylib import myfuncs
from Mylib.myclasses import FeatureColumnsTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler


def create_feature_and_target_transformer(
    categories_for_OrdinalEncoder_dict, feature_cols, target_col, class_names
):
    feature_transformer = ColumnTransformer(
        transformers=[
            (
                "1",
                FeatureColumnsTransformer(
                    categories_for_OrdinalEncoder_dict=categories_for_OrdinalEncoder_dict
                ),
                feature_cols,
            )
        ]
    )

    target_transformer = ColumnTransformer(
        transformers=[("1", OrdinalEncoder(categories=[class_names]), [target_col])]
    )

    return feature_transformer, target_transformer


def transform_data(
    df_train_corrected, df_val_corrected, feature_transformer, target_transformer
):
    # Fit và transform tập train
    df_train_features = feature_transformer.fit_transform(df_train_corrected)
    df_train_target = target_transformer.fit_transform(df_train_corrected)
    df_train_target = df_train_target.reshape(-1)  # Chuyển về 1D array

    # Transform tập val
    df_val_features = feature_transformer.transform(df_val_corrected)
    df_val_target = target_transformer.transform(df_val_corrected)
    df_val_target = df_val_target.reshape(-1)  # Chuyển về 1D array

    # Thay đổi kiểu dữ liệu
    df_train_features = df_train_features.astype("float32")
    df_train_target = df_train_target.astype("int8")
    df_val_features = df_val_features.astype("float32")
    df_val_target = df_val_target.astype("int8")

    return df_train_features, df_train_target, df_val_features, df_val_target
