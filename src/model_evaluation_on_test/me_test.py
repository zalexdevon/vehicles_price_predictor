from Mylib import myfuncs, myclasses


def load_data_for_model_evaluation_on_test(
    test_data_path,
    class_names_path,
    correction_transformer_path,
    data_transformation_path,
    model_path,
):
    test_data = myfuncs.load_python_object(test_data_path)
    class_names = myfuncs.load_python_object(class_names_path)
    correction_transformer = myfuncs.load_python_object(correction_transformer_path)
    feature_transformer = myfuncs.load_python_object(
        f"{data_transformation_path}/feature_transformer.pkl"
    )
    target_transformer = myfuncs.load_python_object(
        f"{data_transformation_path}/target_transformer.pkl"
    )
    model = myfuncs.load_python_object(model_path)

    return (
        test_data,
        class_names,
        correction_transformer,
        feature_transformer,
        target_transformer,
        model,
    )


def transform_test_data(
    test_data, correction_transformer, feature_transformer, target_transformer
):
    # Transform tập test
    df_test_corrected = correction_transformer.transform(test_data)
    df_test_feature = feature_transformer.transform(df_test_corrected)
    df_test_target = target_transformer.transform(df_test_corrected).values.reshape(-1)

    # Thay đổi kiểu dữ liệu
    df_test_feature = df_test_feature.astype("float32")
    df_test_target = df_test_target.astype("int8")

    return df_test_feature, df_test_target


def evaluate_model_on_test(
    test_features,
    test_target,
    model,
    class_names,
    model_evaluation_on_test_path,
):

    final_model_results_text = "===========Kết quả đánh giá model ================\n"

    model_results_text, test_confusion_matrix = myclasses.ClassifierEvaluator(
        model=model,
        class_names=class_names,
        train_feature_data=test_features,
        train_target_data=test_target,
    ).evaluate()
    final_model_results_text += model_results_text

    # Lưu confusion matrix cho tập test
    test_confusion_matrix_path = (
        f"{model_evaluation_on_test_path}/test_confusion_matrix.png"
    )
    test_confusion_matrix.savefig(
        test_confusion_matrix_path, dpi=None, bbox_inches="tight", format=None
    )

    # Lưu vào file results.txt
    with open(f"{model_evaluation_on_test_path}/result.txt", mode="w") as file:
        file.write(final_model_results_text)
