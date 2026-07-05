# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import os
import sys
import hydra
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import tensorflow as tf
import mlflow

from clearml import Task
from clearml.backend_config.defs import get_active_config_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from api.api import get_model
from common.utils import mlflow_ini, set_gpu_memory_limit, get_random_seed, log_to_file
from common.benchmarking import benchmark, cloud_connect
from common.evaluation import gen_load_val
from common.prediction import gen_load_val_predict
from common.quantization import define_extra_options
from face_detection.tf.src.utils import get_config
from face_detection.tf.src.evaluation import evaluate
from face_detection.tf.src.quantization import quantize
from face_detection.tf.src.prediction import predict
from face_detection.tf.src.deployment import deploy



# This function turns Tensorflow's eager mode on and off.
# Eager mode is for debugging the Model Zoo code and is slower.
# Do not set argument to True to avoid runtime penalties.
tf.config.run_functions_eagerly(False)


def process_mode(cfg: DictConfig):
    """
    Execution of the various services

    Args:
        cfg: Configuration dictionary.

    Returns:
        None
    """

    mode = cfg.operation_mode
    mlflow.log_param("model_path", cfg.model.model_path)
    # logging the operation_mode in the output_dir/stm32ai_main.log file
    log_to_file(cfg.output_dir, f'operation_mode: {mode}')

    saved_model_dir = os.path.join(cfg.output_dir, cfg.general.saved_models_dir)
    os.makedirs(saved_model_dir, exist_ok=True)
    model = get_model(cfg=cfg)


    if mode == "evaluation":
        # Generates the model to be loaded on the stm32n6 device using stedgeai core,
        # then loads it and validates in on the device if required.
        gen_load_val(cfg=cfg, model=model)
        # Launches evaluation on the target through the model zoo evaluation service
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        evaluate(cfg, model=model)
        print("[INFO] evaluation complete")

    elif mode == "quantization":
        extra_options = define_extra_options(cfg=cfg)
        quantize(cfg, model=model, extra_options=extra_options)
        print("[INFO] quantization complete")

    elif mode == "prediction":
        # Generates the model to be loaded on the stm32n6 device using stedgeai core,
        # then loads it and validates in on the device if required.
        gen_load_val_predict(cfg=cfg, model=model)
        # Launches prediction on the target through the model zoo prediction service
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        predict(cfg, model=model)
        print("[INFO] prediction complete")

    elif mode == 'benchmarking':
        benchmark(cfg, model_path_to_benchmark=model.model_path)
        print("[INFO] benchmarking complete")

    elif mode == 'deployment':
        deploy(cfg=cfg, model_path_to_deploy=model.model_path)
        print("[INFO] deployment complete")
        if cfg.deployment.hardware_setup.board == "STM32N6570-DK":
            print('[INFO] : Please on STM32N6570-DK toggle the boot switches to the left and power cycle the board.')

    elif mode == 'chain_eqe':
        evaluate(cfg, model=model)
        extra_options = define_extra_options(cfg=cfg)
        quantized_model=quantize(cfg, model=model, extra_options=extra_options)
        evaluate(cfg, model=quantized_model)
        print("Quantized model path:", quantized_model.model_path)
        print("[INFO] chain_eqe complete")

    elif mode == 'chain_eqeb':
        credentials = None
        if cfg.tools.stedgeai.on_cloud:
            _, _, credentials = cloud_connect(stedgeai_core_version=cfg.tools.stedgeai.version)
        evaluate(cfg, model=model)
        extra_options = define_extra_options(cfg=cfg)
        quantized_model = quantize(cfg, model=model, extra_options=extra_options)
        evaluate(cfg, model=quantized_model)
        benchmark(cfg, model_path_to_benchmark=quantized_model.model_path, credentials=credentials)
        print("Quantized model path:", quantized_model.model_path)
        print("[INFO] chain_eqeb complete")

    elif mode == 'chain_qb':
        credentials = None
        if cfg.tools.stedgeai.on_cloud:
            _, _, credentials = cloud_connect(stedgeai_core_version=cfg.tools.stedgeai.version)
        extra_options = define_extra_options(cfg=cfg)
        quantized_model = quantize(cfg, model=model, extra_options=extra_options)
        benchmark(cfg, model_path_to_benchmark=quantized_model.model_path, credentials=credentials)
        print("Quantized model path:", quantized_model.model_path)
        print("[INFO] chain_qb complete")

    elif mode == 'chain_qd':
        extra_options = define_extra_options(cfg=cfg)
        quantized_model = quantize(cfg, model=model, extra_options=extra_options)
        deploy(cfg, model_path_to_deploy=quantized_model.model_path)
        print("Quantized model path:", quantized_model.model_path)
        print("[INFO] chain_qd complete")

    else:
        raise RuntimeError(f"Internal error: invalid operation mode: {mode}")

    if mode in ['benchmarking', 'chain_tbqeb', 'chain_qb', 'chain_eqeb']:
        mlflow.log_param("stedgeai_core_version", cfg.tools.stedgeai.version)
        mlflow.log_param("target", cfg.benchmarking.board)

    # logging the completion of the chain
    log_to_file(cfg.output_dir, f'operation finished: {mode}')

    # ClearML - Example how to get task's context anywhere in the file.
    # Checks if there's a valid ClearML configuration file
    if get_active_config_file() is not None: 
        print(f"[INFO] : ClearML task connection")
        task = Task.current_task()
        task.connect(cfg)


@hydra.main(version_base=None, config_path="", config_name="user_config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point of the script.

    Args:
        cfg: Configuration dictionary.

    Returns:
        None
    """

    # Configure the GPU (the 'general' section may be missing)
    if "general" in cfg and cfg.general:
        # Set upper limit on usable GPU memory
        if "gpu_memory_limit" in cfg.general and cfg.general.gpu_memory_limit:
            set_gpu_memory_limit(cfg.general.gpu_memory_limit)
        else:
            print("[WARNING] The usable GPU memory is unlimited.\n"
                  "Please consider setting the 'gpu_memory_limit' attribute "
                  "in the 'general' section of your configuration file.")

    # Parse the configuration file
    cfg = get_config(cfg)
    cfg.output_dir = HydraConfig.get().runtime.output_dir
    mlflow_ini(cfg)

    # Checks if there's a valid ClearML configuration file
    print(f"[INFO] : ClearML config check")
    if get_active_config_file() is not None:
        print(f"[INFO] : ClearML initialization and configuration")
        # ClearML - Initializing ClearML's Task object.
        task = Task.init(project_name=cfg.general.project_name,
                         task_name='od_modelzoo_task')
        # ClearML - Optional yaml logging 
        task.connect_configuration(name=cfg.operation_mode, 
                                   configuration=cfg)

    # Seed global seed for random generators
    seed = get_random_seed(cfg)
    print(f'[INFO] : The random seed for this simulation is {seed}')
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    # The default hardware type is "MCU".
    process_mode(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='', help='Path to folder containing configuration file')
    parser.add_argument('--config-name', type=str, default='user_config', help='name of the configuration file')

    # Add arguments to the parser
    parser.add_argument('params', nargs='*',
                        help='List of parameters to over-ride in config.yaml')
    args = parser.parse_args()

    # Call the main function
    main()

    # log the config_path and config_name parameters
    mlflow.log_param('config_path', args.config_path)
    mlflow.log_param('config_name', args.config_name)
    mlflow.end_run()
