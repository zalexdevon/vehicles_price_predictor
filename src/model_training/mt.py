from Mylib import myfuncs


def load_data_for_model_training(data_transformation_path):
    train_feature = myfuncs.load_python_object(
        f"{data_transformation_path}/train_features.pkl"
    )
    train_target = myfuncs.load_python_object(
        f"{data_transformation_path}/train_target.pkl"
    )
    val_feature = myfuncs.load_python_object(
        f"{data_transformation_path}/val_features.pkl"
    )
    val_target = myfuncs.load_python_object(
        f"{data_transformation_path}/val_target.pkl"
    )

    return train_feature, train_target, val_feature, val_target
