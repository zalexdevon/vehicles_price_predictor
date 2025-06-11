from Mylib import (
    myfuncs,
    myclasses,
    sk_myfuncs,
    sk_create_model,
)
import numpy as np
import time
import gc
import pandas as pd
from datetime import datetime
import os
from pathlib import Path
from src import const, funcs


class ModelTrainer:
    def __init__(
        self,
        model_training_path,
        num_models,
        scoring,
    ):
        self.model_training_path = model_training_path
        self.num_models = num_models
        self.scoring = scoring

    def train(
        self,
    ):
        # Tạo thư mục lưu kết quả mô hình tốt nhất
        model_training_run_path = Path(
            f"{self.model_training_path}/{self.get_folder_name()}"
        )
        myfuncs.create_directories([model_training_run_path])

        # Get list_param và lưu lại
        list_param = self.get_list_param()
        myfuncs.save_python_object(
            Path(f"{model_training_run_path}/list_param.pkl"), list_param
        )

        # Get các tham số cần thiết khác
        best_val_scoring = -np.inf
        sign_for_val_scoring_find_best_model = (
            self.get_sign_for_val_scoring_to_find_best_model()
        )

        best_model_result_path = Path(f"{model_training_run_path}/best_result.pkl")

        for i, param in enumerate(list_param):
            try:
                # Tạo train, val data
                train_features, train_target, val_features, val_target = (
                    funcs.create_train_val_data(param)
                )

                # Tạo model
                model = funcs.create_model(param)

                # Train model
                print(f"Train model {i} / {self.num_models}")
                model.fit(train_features, train_target)

                train_scoring = sk_myfuncs.evaluate_model_on_one_scoring(
                    model,
                    train_features,
                    train_target,
                    self.scoring,
                )
                val_scoring = sk_myfuncs.evaluate_model_on_one_scoring(
                    model,
                    val_features,
                    val_target,
                    self.scoring,
                )

                # In kết quả
                training_result_text = f"{param}\n -> Val {self.scoring}: {val_scoring}, Train {self.scoring}: {train_scoring}\n"
                print(training_result_text)

                # Cập nhật best model và lưu lại
                val_scoring_find_best_model = (
                    val_scoring * sign_for_val_scoring_find_best_model
                )

                if best_val_scoring < val_scoring_find_best_model:
                    best_val_scoring = val_scoring_find_best_model

                    # Lưu kết quả
                    myfuncs.save_python_object(
                        best_model_result_path,
                        (param, val_scoring, train_scoring),
                    )

                # Giải phóng bộ nhớ

            except:
                continue

        # In ra kết quả của model tốt nhất
        best_model_result = myfuncs.load_python_object(best_model_result_path)
        best_model_result_text = f"Model tốt nhất\n{best_model_result[0]}\n -> Val {self.scoring}: {best_model_result[1]}, Train {self.scoring}: {best_model_result[2]}\n"
        print(best_model_result_text)

    def get_list_param(self):
        # Get full_list_param
        param_dict = myfuncs.load_python_object(
            self.model_training_path / "param_dict.pkl"
        )
        full_list_param = myfuncs.get_full_list_dict(param_dict)

        # Get folder của run
        run_folders = funcs.get_run_folders(self.model_training_path)

        if len(run_folders) > 0:
            # Get list param còn lại
            for run_folder in run_folders:
                list_param = myfuncs.load_python_object(
                    Path(f"{self.model_training_path}/{run_folder}/list_param.pkl")
                )
                full_list_param = myfuncs.subtract_2list_set(
                    full_list_param, list_param
                )

        # Random list
        return myfuncs.randomize_list(full_list_param, self.num_models)

    def get_folder_name(self):
        # Get các folder lưu model tốt nhất
        run_folders = funcs.get_run_folders(self.model_training_path)

        if len(run_folders) == 0:  # Lần đầu tiên chạy thì là run0
            return "run0"

        number_in_run_folders = run_folders.str.extract(r"(\d+)").astype("int")[
            0
        ]  # Các con số trong run0, run1, ... (0, 1, )
        folder_name = f"run{number_in_run_folders.max() +1}"  # Tên folder sẽ là số lớn nhất để prevent trùng
        return folder_name

    def get_sign_for_val_scoring_to_find_best_model(self):
        if self.scoring in const.SCORINGS_PREFER_MININUM:
            return -1

        if self.scoring in const.SCORINGS_PREFER_MAXIMUM:
            return 1

        raise ValueError(f"Chưa định nghĩa cho {self.scoring}")


class BestResultSearcher:
    def __init__(self, model_training_path, scoring):
        self.model_training_path = model_training_path
        self.scoring = scoring

    def next(self):
        run_folders = funcs.get_run_folders(self.model_training_path)
        list_result = []

        for run_folder in run_folders:
            best_result_path = self.model_training_path / run_folder / "best_result.pkl"
            list_result.append(myfuncs.load_python_object(best_result_path))

        best_result = self.get_best_result(list_result)
        return best_result

    def get_best_result(self, list_result):
        list_result = sorted(
            list_result,
            key=lambda item: item[1],
            reverse=self.get_reverse_param_in_sorted(),
        )  # Sort theo val scoring
        return list_result[0]

    def get_reverse_param_in_sorted(self):
        if self.scoring in const.SCORINGS_PREFER_MAXIMUM:
            return True

        if self.scoring in const.SCORINGS_PREFER_MININUM:
            return False

        raise ValueError(f"Chưa định nghĩa cho {self.scoring}")


class ModelRetrainer:
    def __init__(self, train_features, train_target, best_param):
        self.train_features = train_features
        self.train_target = train_target
        self.best_param = best_param

    def next(self):
        # Tạo model
        model = funcs.create_model(self.best_param)

        # Train model
        model.fit(self.train_features, self.train_target)

        return model


class ModelEvaluator:
    def __init__(
        self, val_features, val_target, class_names, model, model_evaluation_on_val_path
    ):
        self.val_features = val_features
        self.val_target = val_target
        self.class_names = class_names
        self.model = model
        self.model_evaluation_on_val_path = model_evaluation_on_val_path

    def next(self):
        # Đánh giá
        model_result_text = "===============Kết quả đánh giá model==================\n"

        # Đánh giá model trên tập val
        result_text, val_confusion_matrix = myclasses.ClassifierEvaluator(
            model=self.model,
            class_names=self.class_names,
            train_feature_data=self.val_features,
            train_target_data=self.val_target,
        ).evaluate()
        model_result_text += result_text  # Thêm đoạn đánh giá vào

        # Lưu lại confusion matrix cho tập val
        val_confusion_matrix_path = Path(
            f"{self.model_evaluation_on_val_path}/confusion_matrix.png"
        )
        val_confusion_matrix.savefig(
            val_confusion_matrix_path, dpi=None, bbox_inches="tight", format=None
        )

        # Lưu vào file results.txt
        with open(f"{self.model_evaluation_on_val_path}/result.txt", mode="w") as file:
            file.write(model_result_text)


class ModelTrainingResultGatherer:
    MODEL_TRAINING_FOLDER_PATH = "artifacts/model_training"

    def __init__(self, scoring):
        self.scoring = scoring
        pass

    def next(self):
        model_training_paths = [
            f"{self.MODEL_TRAINING_FOLDER_PATH}/{item}"
            for item in os.listdir(self.MODEL_TRAINING_FOLDER_PATH)
        ]

        result = []
        for folder_path in model_training_paths:
            result += self.get_result_from_1folder(folder_path)

        # Sort theo val_scoring (ở vị trí thứ 1)
        result = sorted(
            result,
            key=lambda item: item[1],
            reverse=funcs.get_reverse_param_in_sorted(self.scoring),
        )
        return result

    def get_result_from_1folder(self, folder_path):
        run_folder_names = funcs.get_run_folders(folder_path)
        run_folder_paths = [f"{folder_path}/{item}" for item in run_folder_names]

        list_result = []
        for folder_path in run_folder_paths:
            result = myfuncs.load_python_object(f"{folder_path}/result.pkl")
            list_result.append(result)

        return list_result
