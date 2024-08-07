import os
import subprocess as sp
import pandas as pd

# to use this, just run python run_batch_job.py in cmd

gpu = "0.1" # amount of GPU to use per task
job_name = "mustafa-test-batch-job"

base = "/home/runner/talmodata-smb/mustafa/dreem-experiments/run/mice-btc" #where to run the job from
dreem_repo = "/home/runner/talmodata-smb/mustafa/dreem-experiments/src/dreem" #where the dreem repo is stored

config_dir=os.path.join(base, "configs") #where to find the configs
config_name= "base" #base config name
params_cfg = os.path.join(config_dir, "override.yaml") #override config 

# if running just 1 job, comment this line out and delete the ++batch_config command in the command below
# each row in this file is a separate run with overrides
# naming method: have the first column as logging.name (wandb logging); this creates the directory ./models/logging.name
task_csv = os.path.join(config_dir, "demo_batch.csv") # csv for tasks - each pod is a task

# number of VMs that are spun up (also the number of tasks that you are running)
# note that the server must be mounted locally as a network location to use this if the csv is on the cluster
pods = len(pd.read_csv(task_csv.replace("/home/runner/talmodata-smb", "/Volumes/talmodata")))
par = min(int(1/float(gpu)), pods) #number of tasks that can be run in parallel (always smaller than pods)

# enter your WANDB API KEY in the cmd section
# mount both smb and vast volumes
cmd = [
    "runai",
    "submit",
    "--gpu",
    gpu,
    "--name",
    job_name,
    "--preemptible",
    "-i",
    "asheridan/biogtr",
    "-v",
    "/data/talmolab-smb:/home/runner/talmodata-smb", 
    "-v",
    "/talmo:/home/runner/vast"
    "-e",
    f"RUNNER_CMD=cp -r {dreem_repo} ~ && mamba env create -n dreem -f ~/dreem/environment.yml && export WANDB_API_KEY=6cc5012a6ecfb9cd970bd07686dbfcefd3190a04 && cd {base} && conda run -n dreem dreem-train --config-dir={config_dir} --config-name={config_name} ++params_config={params_cfg} ++batch_config={task_csv}",
    "--parallelism",
    str(par),
    "--completions",
    str(pods),
]

print(f"base directory: {base}")
print(f"running with {pods} pods")
print(f"max pods that can run concurrently: {par}")
print(f"runner command: {cmd}")

sp.run(cmd)