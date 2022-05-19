import os

models = ["dt", "rf", "nn_1bhl_2", "nn_1bhl_4", "nn_1shl_2", "nn_1shl_4", "nn_steps", "svc"]
script_path = "/home/hanchard/benchmarking/benchmarking_batch_burger_best.sh"

for model in models:
        for step in ["f", 's']:
                for num in [1, 2]:
                        if "nn" in model:
                                sc = "nn"
                        else:
                                sc = "ml"
                        cmd = f"qsub -v PREFIX={'BURG'},MODEL={model},SCRIPT={sc},STEP={step},NUM={num} -N burg_best_{model} -o pbs_out_bm_burg_best_{model}_{step}_{num}.dat -e pbs_err_bm_burg_best_{model}_{step}_{num}.dat {script_path}"
                        os.system(cmd)
