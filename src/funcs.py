from Mylib import myfuncs, sk_create_model, sk_myfuncs
from Mylib.myclasses import FeatureColumnsTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import pandas as pd
import os


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


def create_model(param):
    model_name = param.pop("model_name")
    param.pop("list_after_transformer")

    return sk_create_model.create_model(model_name, param)


def create_train_val_data(param):
    train_features = myfuncs.load_python_object(
        param["train_val_path"] / "train_features.pkl"
    )
    train_target = myfuncs.load_python_object(
        param["train_val_path"] / "train_target.pkl"
    )
    val_features = myfuncs.load_python_object(
        param["train_val_path"] / "val_features.pkl"
    )
    val_target = myfuncs.load_python_object(param["train_val_path"] / "val_target.pkl")

    after_transformer = sk_myfuncs.convert_list_estimator_into_pipeline(
        param["list_after_transformer"]
    )
    train_features = after_transformer.fit_transform(train_features)
    val_features = after_transformer.transform(val_features)

    return train_features, train_target, val_features, val_target


def get_run_folders(model_training_path):
    run_folders = pd.Series(os.listdir(model_training_path))
    run_folders = run_folders[run_folders.str.startswith("run")]
    return run_folders


def create_train_test_data(param, df_test):
    # Load data
    correction_transformer = myfuncs.load_python_object(
        param["train_val_path"] / "correction_transformer.pkl"
    )
    feature_transformer = myfuncs.load_python_object(
        param["train_val_path"] / "feature_transformer.pkl"
    )
    target_transformer = myfuncs.load_python_object(
        param["train_val_path"] / "target_transformer.pkl"
    )
    train_features = myfuncs.load_python_object(
        param["train_val_path"] / "train_features.pkl"
    )
    train_target = myfuncs.load_python_object(
        param["train_val_path"] / "train_target.pkl"
    )

    # Transform tập test
    df_test_corrected = correction_transformer.transform(df_test)
    test_features = feature_transformer.transform(df_test_corrected)
    test_target = target_transformer.transform(df_test_corrected).reshape(-1)

    after_transformer = sk_myfuncs.convert_list_estimator_into_pipeline(
        param["list_after_transformer"]
    )

    train_features = after_transformer.fit_transform(train_features)
    test_features = after_transformer.transform(test_features)

    return train_features, train_target, test_features, test_target
