import os

models = ["dt", "rf", "nn_1bhl_2", "nn_1bhl_4", "nn_1shl_2", "nn_1shl_4", "nn_steps", "svc"]
script_path = "/home/hanchard/benchmarking/benchmarking_batch.sh"

for model in models:
    for prefix in ["balanced", "undersampled"]:
        if "nn" in model:
                sc = "nn"
        else:
                sc = "ml"
        cmd = f"qsub -v PREFIX={prefix},MODEL={model},SCRIPT={sc} -N bm_{prefix[:3]}_{model} -o pbs_out_BATCH_full_{prefix[:3]}_{model}.dat -e pbs_err_BATCH_full_{prefix[:3]}_{model}.dat {script_path}"
        os.system(cmd)
