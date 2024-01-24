import datetime
import gc
import importlib
import itertools
import json
import os
import struct
import time
from typing import Any, Dict, Optional, Tuple

from absl import app
from absl import flags
from absl import logging
import torch
import torch.distributed as dist
from algorithmic_efficiency.workloads.sennet.sennet_pytorch.workload import SennetWorkload

from algorithmic_efficiency import checkpoint_utils
from algorithmic_efficiency import halton
from algorithmic_efficiency import logger_utils
from algorithmic_efficiency import random_utils as prng
from algorithmic_efficiency import spec
from algorithmic_efficiency.profiler import PassThroughProfiler
from algorithmic_efficiency.profiler import Profiler
from algorithmic_efficiency.pytorch_utils import pytorch_init
from algorithmic_efficiency.pytorch_utils import pytorch_setup
from algorithmic_efficiency.pytorch_utils import sync_ddp_time
from algorithmic_efficiency.workloads import workloads

flags.DEFINE_string(
    "submission_path",
    None,
    "The relative path of the Python file containing submission functions. "
    "NOTE: the submission dir must have an __init__.py file!",
)
flags.DEFINE_string(
    "workload",
    None,
    "The name of the workload to run."
)
flags.DEFINE_string("tuning_search_space", None, "The path to the JSON file describing the external tuning search space.")
flags.DEFINE_integer("num_tuning_trials", 1, "The number of external hyperparameter trials to run.")
flags.DEFINE_string("data_dir", None, "Dataset location.")
flags.DEFINE_boolean(
    "torch_compile",
    True,
    "Whether to use `torch.compile` to JIT-compile PyTorch code. " "This will only take effect when `framework`==pytorch.",
)
flags.DEFINE_string(
    "experiment_dir",
    None,
    "The root directory to store all experiments. "
    "It is required and the directory should have "
    "an absolute path rather than a relative path.",
)
flags.DEFINE_string("experiment_name", None, "Name of the experiment.")
flags.DEFINE_boolean(
    "save_intermediate_checkpoints",
    True,
    "Whether to save any intermediate checkpoints. " "If False, it will only keep the latest checkpoint.",
)
flags.DEFINE_boolean("resume_last_run", None, "Whether to resume the experiment from its last run.")
flags.DEFINE_boolean(
    "append_timestamp",
    False,
    "If True, the current datetime will be appended to the experiment name. "
    "Useful for guaranteeing a unique experiment dir for new runs.",
)
flags.DEFINE_boolean("use_wandb", False, "Whether to use Weights & Biases logging.")
flags.DEFINE_boolean("profile", False, "Whether to produce profiling output.")
flags.DEFINE_integer("max_global_steps", None, "Maximum number of update steps.")
flags.DEFINE_boolean(
    "overwrite", False, "Whether to overwrite the experiment with identical experiment_dir and experiment_name."
)
flags.DEFINE_boolean("save_checkpoints", True, "Whether or not to checkpoint the model at every eval.")
flags.DEFINE_integer("hparam_start_index", None, "Start index to slice set of hyperparameters in tuning search space.")
flags.DEFINE_integer("hparam_end_index", None, "End index to slice set of hyperparameters in tuning spearch space.")
flags.DEFINE_integer("rng_seed", None, "Value of rng seed. If None, a random seed will" "be generated from hardware.")
flags.DEFINE_boolean("set_pytorch_max_split_size", False, "If true, set pytorch max_split_size_mb to 256")
FLAGS = flags.FLAGS
USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()


def _get_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def _get_time_ddp():
    torch.cuda.synchronize()
    t = time.time()
    return sync_ddp_time(t, DEVICE)


if USE_PYTORCH_DDP:
    get_time = _get_time_ddp
else:
    get_time = _get_time


def _reset_cuda_mem():
    if torch.cuda.is_available():
        torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def train_once(
    workload: spec.Workload,
    workload_name: str,
    global_batch_size: int,
    global_eval_batch_size: int,
    data_dir: str,
    imagenet_v2_data_dir: str,
    init_optimizer_state: spec.InitOptimizerFn,
    update_params: spec.UpdateParamsFn,
    data_selection: spec.DataSelectionFn,
    hyperparameters: Optional[spec.Hyperparameters],
    rng_seed: int,
    rng: spec.RandomState,
    profiler: Profiler,
    max_global_steps: int = None,
    log_dir: Optional[str] = None,
    save_checkpoints: Optional[bool] = True,
) -> Tuple[spec.Timing, Dict[str, Any]]:
    data_rng, opt_init_rng, model_init_rng, rng = prng.split(rng, 4)

    # Workload setup.
    logging.info("Initializing dataset.")
    with profiler.profile("Initializing dataset"):
        input_queue = workload._build_input_queue(data_rng, "train", data_dir=data_dir, global_batch_size=global_batch_size)
    logging.info("Initializing model.")
    with profiler.profile("Initializing model"):
        dropout_rate = None
        aux_dropout_rate = None
        if hasattr(hyperparameters, "dropout_rate"):
            dropout_rate = hyperparameters.dropout_rate
        if hasattr(hyperparameters, "aux_dropout_rate"):
            aux_dropout_rate = hyperparameters.aux_dropout_rate
        model_params, model_state = workload.init_model_fn(model_init_rng, dropout_rate, aux_dropout_rate)
    logging.info("Initializing optimizer.")
    with profiler.profile("Initializing optimizer"):
        optimizer_state = init_optimizer_state(workload, model_params, model_state, hyperparameters, opt_init_rng)

    logging.info("Initializing metrics bundle.")
    # Bookkeeping.
    train_state = {
        "validation_goal_reached": False,
        "test_goal_reached": False,
        "is_time_remaining": True,
        "last_eval_time": 0,
        "training_complete": False,
        "accumulated_submission_time": 0,
        "accumulated_eval_time": 0,
        "accumulated_logging_time": 0,
        "last_step_end_time": None,
    }
    global_step = 0
    eval_results = []
    preemption_count = 0

    # Loggers and checkpoint setup.
    logging.info("Initializing checkpoint and logger.")
    if log_dir is not None:
        # If the checkpoint exists, load from the checkpoint.
        (
            optimizer_state,
            model_params,
            model_state,
            train_state,
            eval_results,
            global_step,
            preemption_count,
        ) = checkpoint_utils.maybe_restore_checkpoint(
            FLAGS.framework,
            optimizer_state,
            model_params,
            model_state,
            train_state,
            eval_results,
            global_step,
            preemption_count,
            checkpoint_dir=log_dir,
        )
        meta_file_name = os.path.join(log_dir, f"meta_data_{preemption_count}.json")
        logging.info(f"Saving meta data to {meta_file_name}.")
        meta_data = logger_utils.get_meta_data(workload, rng_seed)
        logger_utils.write_json(meta_file_name, meta_data)
        flag_file_name = os.path.join(log_dir, f"flags_{preemption_count}.json")
        logging.info(f"Saving flags to {flag_file_name}.")
        logger_utils.write_json(flag_file_name, flags.FLAGS.flag_values_dict())
        metrics_logger = logger_utils.set_up_loggers(log_dir, flags.FLAGS, hyperparameters)
        workload.attach_metrics_logger(metrics_logger)

    global_start_time = get_time()
    train_state["last_step_end_time"] = global_start_time

    logging.info("Starting training loop.")
    while train_state["is_time_remaining"] and not train_state["training_complete"]:
        step_rng = prng.fold_in(rng, global_step)
        data_select_rng, update_rng, eval_rng = prng.split(step_rng, 3)

        with profiler.profile("Data selection"):
            batch = data_selection(
                workload, input_queue, optimizer_state, model_params, model_state, hyperparameters, global_step, data_select_rng
            )
        try:
            with profiler.profile("Update parameters"):
                optimizer_state, model_params, model_state = update_params(
                    workload=workload,
                    current_param_container=model_params,
                    current_params_types=workload.model_params_types,
                    model_state=model_state,
                    hyperparameters=hyperparameters,
                    batch=batch,
                    loss_type=workload.loss_type,
                    optimizer_state=optimizer_state,
                    eval_results=eval_results,
                    global_step=global_step,
                    rng=update_rng,
                )
        except spec.TrainingCompleteError:
            train_state["training_complete"] = True
        global_step += 1
        if (max_global_steps is not None) and (global_step == max_global_steps):
            train_state["training_complete"] = True

        train_step_end_time = get_time()
        train_state["accumulated_submission_time"] += train_step_end_time - train_state["last_step_end_time"]
        max_allowed_runtime_sec = workload.max_allowed_runtime_sec
        train_state["is_time_remaining"] = train_state["accumulated_submission_time"] < max_allowed_runtime_sec

        # Check if submission is eligible for an untimed eval.
        time_since_last_eval = train_step_end_time - train_state["last_eval_time"]
        if time_since_last_eval >= workload.eval_period_time_sec or train_state["training_complete"]:
            with profiler.profile("Evaluation"):
                del batch
                _reset_cuda_mem()
                try:
                    eval_start_time = get_time()
                    latest_eval_result = workload.eval_model(
                        global_eval_batch_size, model_params, model_state, eval_rng, data_dir, imagenet_v2_data_dir, global_step
                    )
                    # Save last eval time.
                    eval_end_time = get_time()
                    train_state["last_eval_time"] = eval_end_time

                    # Accumulate eval time.
                    train_state["accumulated_eval_time"] += eval_end_time - eval_start_time

                    # Add times to eval results for logging.
                    latest_eval_result["score"] = train_state["accumulated_submission_time"]
                    latest_eval_result["total_duration"] = eval_end_time - global_start_time
                    latest_eval_result["accumulated_submission_time"] = train_state["accumulated_submission_time"]
                    latest_eval_result["accumulated_eval_time"] = train_state["accumulated_eval_time"]
                    latest_eval_result["accumulated_logging_time"] = train_state["accumulated_logging_time"]
                    time_since_start = latest_eval_result["total_duration"]
                    logging.info(f"Time since start: {time_since_start:.2f}s, \tStep: {global_step}, \t{latest_eval_result}")
                    eval_results.append((global_step, latest_eval_result))

                    logging_start_time = get_time()

                    if log_dir is not None:
                        metrics_logger.append_scalar_metrics(
                            latest_eval_result,
                            global_step=global_step,
                            preemption_count=preemption_count,
                            is_eval=True,
                        )
                        if save_checkpoints:
                            checkpoint_utils.save_checkpoint(
                                framework=FLAGS.framework,
                                optimizer_state=optimizer_state,
                                model_params=model_params,
                                model_state=model_state,
                                train_state=train_state,
                                eval_results=eval_results,
                                global_step=global_step,
                                preemption_count=preemption_count,
                                checkpoint_dir=log_dir,
                                save_intermediate_checkpoints=FLAGS.save_intermediate_checkpoints,
                            )

                    logging_end_time = get_time()
                    train_state["accumulated_logging_time"] += logging_end_time - logging_start_time

                    _reset_cuda_mem()

                except RuntimeError as e:
                    logging.exception(f"Eval step {global_step} error.\n")
                    if "out of memory" in str(e):
                        logging.warning("Error: GPU out of memory during eval during step " f"{global_step}, error : {str(e)}.")
                        _reset_cuda_mem()

        train_state["last_step_end_time"] = get_time()

    metrics = {"eval_results": eval_results, "global_step": global_step}

    if log_dir is not None:
        metrics_logger.append_scalar_metrics(
            {"score": train_state["accumulated_submission_time"]}, global_step=global_step, preemption_count=preemption_count
        )
        metrics_logger.finish()
        checkpoint_utils.save_checkpoint(
            framework=FLAGS.framework,
            optimizer_state=optimizer_state,
            model_params=model_params,
            model_state=model_state,
            train_state=train_state,
            eval_results=eval_results,
            global_step=global_step,
            preemption_count=preemption_count,
            checkpoint_dir=log_dir,
            save_intermediate_checkpoints=FLAGS.save_intermediate_checkpoints,
        )

    return train_state["accumulated_submission_time"], metrics


def score_submission_on_workload(
    workload: spec.Workload,
    workload_name: str,
    submission_path: str,
    data_dir: str,
    tuning_ruleset: str,
    profiler: Optional[Profiler] = None,
    max_global_steps: Optional[int] = None,
    imagenet_v2_data_dir: Optional[str] = None,
    tuning_search_space: Optional[str] = None,
    num_tuning_trials: Optional[int] = None,
    log_dir: Optional[str] = None,
    save_checkpoints: Optional[bool] = True,
    hparam_start_index: Optional[bool] = None,
    hparam_end_index: Optional[bool] = None,
    rng_seed: Optional[int] = None,
):
    # Expand paths because '~' may not be recognized
    data_dir = os.path.expanduser(data_dir)
    # Remove the trailing '.py' and convert the filepath to a Python module.
    submission_module_path = workloads.convert_filepath_to_module(submission_path)
    submission_module = importlib.import_module(submission_module_path)

    init_optimizer_state = submission_module.init_optimizer_state
    update_params = submission_module.update_params
    data_selection = submission_module.data_selection
    global_batch_size = submission_module.get_batch_size(workload_name)

    n_gpus = N_GPUS
    if global_batch_size % n_gpus != 0:
        raise ValueError(
            f"The global batch size ({global_batch_size}) has to be divisible by " f"the number of GPUs ({n_gpus})."
        )
    global_eval_batch_size = workload.eval_batch_size
    if global_eval_batch_size % n_gpus != 0:
        raise ValueError(
            f"The global eval batch size ({global_eval_batch_size}) has to be " f"divisible by the number of GPUs ({n_gpus})."
        )

    if tuning_search_space is None:
        raise ValueError("Must provide a tuning search space JSON file when using external tuning.")
    with open(tuning_search_space, "r", encoding="UTF-8") as search_space_file:
        tuning_search_space = halton.generate_search(json.load(search_space_file), num_tuning_trials)
    all_timings = []
    all_metrics = []
    tuning_search_space_iter = itertools.islice(enumerate(tuning_search_space), hparam_start_index, hparam_end_index)
    for hi, hyperparameters in tuning_search_space_iter:
        # Generate a new seed from hardware sources of randomness for each trial.
        if not rng_seed:
            rng_seed = struct.unpack("I", os.urandom(4))[0]
        logging.info("Using RNG seed %d", rng_seed)
        rng = prng.PRNGKey(rng_seed)
        rng, _ = prng.split(rng, 2)
        logging.info(f"--- Tuning run {hi + 1}/{num_tuning_trials} ---")

        tuning_dir_name = None
        if log_dir is not None:
            tuning_dir_name = os.path.join(log_dir, f"trial_{hi + 1}")
            logging.info(f"Creating tuning directory at {tuning_dir_name}.")
            logger_utils.makedir(tuning_dir_name)
            # If existing hyperparameter exists, use saved
            # hyperparameters for consistency.
            hyperparameters = logger_utils.write_hparams(hyperparameters, tuning_dir_name)
            tuning_search_space[hi] = hyperparameters

        with profiler.profile("Train"):
            imagenet_v2_data_dir = None
            timing, metrics = train_once(
                workload,
                workload_name,
                global_batch_size,
                global_eval_batch_size,
                data_dir,
                imagenet_v2_data_dir,
                init_optimizer_state,
                update_params,
                data_selection,
                hyperparameters,
                rng_seed,
                rng,
                profiler,
                max_global_steps,
                tuning_dir_name,
                save_checkpoints=save_checkpoints,
            )
        all_timings.append(timing)
        all_metrics.append(metrics)
        logging.info(f"Tuning trial {hi + 1}/{num_tuning_trials}")
        logging.info(f"Hyperparameters: {tuning_search_space[hi]}")
        logging.info(f"Metrics: {all_metrics[hi]}")
        logging.info(f"Timing: {all_timings[hi]}")
        num_evals = len(all_metrics[hi]["eval_results"])
        logging.info(f"Total number of evals: {num_evals}")
        logging.info("=" * 20)
    score = min(all_timings)
    return score


def main(_):
    if FLAGS.profile:
        profiler = Profiler()
    else:
        profiler = PassThroughProfiler()

    pytorch_init(USE_PYTORCH_DDP, RANK, profiler)

    workload = SennetWorkload()

    experiment_name = FLAGS.experiment_name
    if experiment_name and FLAGS.append_timestamp:
        experiment_name += datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M-%S")
    logging_dir_path = logger_utils.get_log_dir(
        FLAGS.experiment_dir, FLAGS.workload, "pytorch", experiment_name, FLAGS.resume_last_run, FLAGS.overwrite
    )

    score = score_submission_on_workload(
        workload=workload,
        workload_name=FLAGS.workload,
        submission_path=FLAGS.submission_path,
        data_dir=FLAGS.data_dir,
        tuning_ruleset="external_tuning",
        profiler=profiler,
        max_global_steps=FLAGS.max_global_steps,
        imagenet_v2_data_dir=None,
        tuning_search_space=FLAGS.tuning_search_space,
        num_tuning_trials=FLAGS.num_tuning_trials,
        log_dir=logging_dir_path,
        save_checkpoints=FLAGS.save_checkpoints,
        hparam_start_index=FLAGS.hparam_start_index,
        hparam_end_index=FLAGS.hparam_end_index,
        rng_seed=FLAGS.rng_seed,
    )
    logging.info(f"Final {FLAGS.workload} score: {score}")

    if FLAGS.profile:
        logging.info(profiler.summary())

    if USE_PYTORCH_DDP:
        # Cleanup.
        dist.destroy_process_group()


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("submission_path")
    flags.mark_flag_as_required("experiment_dir")
    flags.mark_flag_as_required("experiment_name")
    app.run(main)
