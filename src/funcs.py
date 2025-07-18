from Mylib import myfuncs, sk_myfuncs, sk_create_object
from Mylib.myclasses import FeatureColumnsTransformer
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import os
from src import const


def create_feature_and_target_transformer(
    categories_for_OrdinalEncoder_dict, class_names
):
    feature_transformer = FeatureColumnsTransformer(
        categories_for_OrdinalEncoder_dict=categories_for_OrdinalEncoder_dict
    )

    target_transformer = OrdinalEncoder(categories=[class_names])

    return feature_transformer, target_transformer


def transform_data(
    df_train_corrected,
    df_val_corrected,
    feature_transformer,
    target_transformer,
):
    # Get cột mục tiêu
    target_col = myfuncs.get_target_col_from_df(df_train_corrected)

    # Fit và transform tập train
    df_train_target = df_train_corrected[[target_col]]
    df_train_corrected = df_train_corrected.drop(columns=[target_col])
    df_train_corrected = feature_transformer.fit_transform(df_train_corrected).astype(
        "float32"
    )
    df_train_target = (
        target_transformer.fit_transform(df_train_target).reshape(-1).astype("int8")
    )

    # Transform tập val
    df_val_target = df_val_corrected[[target_col]]
    df_val_corrected = df_val_corrected.drop(columns=[target_col])
    df_val_corrected = feature_transformer.transform(df_val_corrected).astype("float32")
    df_val_target = (
        target_transformer.transform(df_val_target).reshape(-1).astype("int8")
    )

    return df_train_corrected, df_train_target, df_val_corrected, df_val_target


def create_model(param):
    model = sk_create_object.ObjectCreatorFromDict(param, "model").next()

    return model


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


def get_reverse_param_in_sorted(scoring):
    if scoring in const.SCORINGS_PREFER_MAXIMUM:
        return True

    if scoring in const.SCORINGS_PREFER_MININUM:
        return False

    raise ValueError(f"Chưa định nghĩa cho {scoring}")
