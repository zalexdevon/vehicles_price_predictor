from Mylib import myfuncs, myclasses


def load_data_for_me_trainval(data_transformation_path, model_path, class_names_path):
    train_features = myfuncs.load_python_object(
        f"{data_transformation_path}/train_features.pkl"
    )
    train_target = myfuncs.load_python_object(
        f"{data_transformation_path}/train_target.pkl"
    )
    val_features = myfuncs.load_python_object(
        f"{data_transformation_path}/val_features.pkl"
    )
    val_target = myfuncs.load_python_object(
        f"{data_transformation_path}/val_target.pkl"
    )

    model = myfuncs.load_python_object(model_path)
    class_names = myfuncs.load_python_object(class_names_path)

    return train_features, train_target, val_features, val_target, model, class_names


def evaluate_model_on_train_val(
    train_features,
    train_target,
    val_features,
    val_target,
    model,
    class_names,
    model_evaluation_on_train_val_path,
):
    final_model_results_text = (
        "===============Kết quả đánh giá model==================\n"
    )

    # Đánh giá model trên tập train, val
    model_results_text, train_confusion_matrix, val_confusion_matrix = (
        myclasses.ClassifierEvaluator(
            model=model,
            class_names=class_names,
            train_feature_data=train_features,
            train_target_data=train_target,
            val_feature_data=val_features,
            val_target_data=val_target,
        ).evaluate()
    )
    final_model_results_text += model_results_text  # Thêm đoạn đánh giá vào

    # Lưu lại confusion matrix cho tập train và val
    train_confusion_matrix_path = (
        f"{model_evaluation_on_train_val_path}/train_confusion_matrix.png"
    )
    train_confusion_matrix.savefig(
        train_confusion_matrix_path, dpi=None, bbox_inches="tight", format=None
    )
    val_confusion_matrix_path = (
        f"{model_evaluation_on_train_val_path}/val_confusion_matrix.png"
    )
    val_confusion_matrix.savefig(
        val_confusion_matrix_path, dpi=None, bbox_inches="tight", format=None
    )

    # Lưu vào file results.txt
    with open(f"{model_evaluation_on_train_val_path}/result.txt", mode="w") as file:
        file.write(final_model_results_text)
