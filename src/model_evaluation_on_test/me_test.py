from Mylib import myfuncs, myclasses


def transform_test_data(
    test_data, correction_transformer, feature_transformer, target_transformer
):
    # Transform tập test
    df_test_corrected = correction_transformer.transform(test_data)
    df_test_feature = feature_transformer.transform(df_test_corrected)
    df_test_target = target_transformer.transform(df_test_corrected).reshape(-1)

    # Thay đổi kiểu dữ liệu
    df_test_feature = df_test_feature.astype("float32")
    df_test_target = df_test_target.astype("int8")

    return df_test_feature, df_test_target
