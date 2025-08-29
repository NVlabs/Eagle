from one_logger_utils.huggingface import TimeEventCallback, hook_trainer_cls
import os
from transformers import Trainer

def create_onelogger_config(training_args, data_args):
    
    app_tag = data_args.meta_path.split('/')[-2]
    one_logger_callback_config  = {
        "enable_for_current_rank": data_args.use_onelogger and os.environ.get('RANK') == '0',                        # OneLogger will only log metrics on a single rank. This flag specifies which rank to use for metric logging.
        "one_logger_async": True,                                                        # If you already use wandbLogger in your code, please set this as true.
        "one_logger_project": "eagle_commercial",                                           # Define your poject name
        "log_every_n_train_iterations": 10, # The log interval for logging metrics during training.
        "app_tag_run_version": "0.0.0",                                                  # The model/application version
        "summary_data_schema_version": "1.0.0",                                          # Data schema version. Typically should not be modified.
        "app_run_type": "training",                                                      # Currently, training is the only supported job type.
        "app_tag": app_tag,                                              # Please change this! This is used for performance tracking -- Any job which is expected to have the same training performance, should have the same app_tag
        "app_tag_run_name": app_tag,                                           # Please change this! This is used for tracking OVERALL progress for a training across multiple jobs. Jobs with same model are supposed to have the same value.
        "world_size": int(os.environ.get('WORLD_SIZE', 1)),
        "global_batch_size": training_args.per_device_train_batch_size * int(os.environ.get('WORLD_SIZE', 1)),
        "batch_size": training_args.per_device_train_batch_size,
        "train_iterations_target": 10000,
        "train_samples_target": 160000,
        "is_train_iterations_enabled": True,                                             # Does this job include train iterations.
        "is_baseline_run": False,                                                        # Is this job used for baseline metrics calculating. It should be marked as False for most case.
        "is_test_iterations_enabled": False,                                             # Does this job include test iterations
        "is_validation_iterations_enabled": True,                                        # Does this job include validataion iterations
        "is_save_checkpoint_enabled": True,                                              # Determines if the job will save checkpoints.
        "is_log_throughput_enabled": False,                                              # Determines if the job includs TFLOPS tracking&calculation
        "micro_batch_size": training_args.per_device_train_batch_size,
        "seq_length": data_args.max_seq_length,
        "save_checkpoint_strategy": "sync",                                              # Specifies sync/synchronous or async/asynchronous checkpoint saving.
    }
    one_logger_callback_utils = TimeEventCallback(one_logger_callback_config)
    return one_logger_callback_utils

def warp_onelogger_trainer(one_logger_callback_utils):
    
    CustomizedTrainer = hook_trainer_cls(Trainer, one_logger_callback_utils=one_logger_callback_utils)
    return CustomizedTrainer