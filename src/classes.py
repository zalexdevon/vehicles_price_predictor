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
        # Khởi tạo biến cho logging
        log_message = ""

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

        # Logging
        log_message += f"Kết quả model tốt nhất lưu tại {model_training_run_path}\n"
        log_message += f"Kết quả train từng model:\n"

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

                # Logging
                log_message += training_result_text

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

        # Logging
        log_message += best_model_result_text

        return log_message

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
        log_message = ""

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

        # Logging
        log_message += model_result_text

        return log_message


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


class LoggingDisplayer:
    DATE_FORMAT = "%d-%m-%Y-%H-%M-%S"
    READ_FOLDER_NAME = "artifacts/logs"
    WRITE_FOLDER_NAME = "artifacts/gather_logs"

    # Tạo thư mục
    os.makedirs(WRITE_FOLDER_NAME, exist_ok=True)

    def __init__(self, mode, file_name=None, start_time=None, end_time=None):
        self.mode = mode
        self.file_name = file_name
        self.start_time = start_time
        self.end_time = end_time

        if self.file_name is None:
            self.file_name = f"{datetime.now().strftime(self.DATE_FORMAT)}.log"

    def print_and_save(self):
        file_path = f"{self.WRITE_FOLDER_NAME}/{self.file_name}"

        if self.mode == "all":
            result = self.gather_all_logging_result()
        else:
            result = self.gather_logging_result_from_start_to_end_time()

        print(result)
        print(f"Lưu result tại {file_path}")
        myfuncs.write_content_to_file(result, file_path)

    def gather_all_logging_result(self):
        logs_filenames = self.get_logs_filenames()

        return self.read_from_logs_filenames(logs_filenames)

    def gather_logging_result_from_start_to_end_time(self):
        logs_filenames = pd.Series(self.get_logs_filenames())
        logs_filenames = logs_filenames[
            (logs_filenames > self.start_time) & (logs_filenames < self.end_time)
        ].tolist()

        return self.read_from_logs_filenames(logs_filenames)

    def read_from_logs_filenames(self, logs_filenames):
        result = ""
        for logs_filename in logs_filenames:
            logs_filepath = f"{self.READ_FOLDER_NAME}/{logs_filename}.log"
            content = myfuncs.read_content_from_file(logs_filepath)
            result += f"{content}\n\n"

        return result

    def get_logs_filenames(self):
        logs_filenames = os.listdir(self.READ_FOLDER_NAME)
        date_format_in_filename = f"{self.DATE_FORMAT}.log"
        logs_filenames = [
            datetime.strptime(item, date_format_in_filename) for item in logs_filenames
        ]
        logs_filenames = sorted(logs_filenames)  # Sắp xếp theo thời gian tăng dần
        return logs_filenames
