import os
import subprocess as sp

gpu = "0.1"
job_name = "mustafa-test-batch-job"

base = "/home/runner/talmodata-smb/aadi/biogtr_expts/run/animal/eight_flies" #where to run the job from
dreem_repo = base.replace("biogtr_expts/run/animal/eight_flies", "dreem") #where the dreem repo is stored

config_dir=os.path.join(base, "configs") #where to find the configs
config_name= "base" #base config name
params_cfg = os.path.join(config_dir, "sample_efficiency.yaml") #override config 
# if running just 1 job, comment this line out and delete the ++batch_config command in the command below
task_csv = os.path.join(config_dir, "sample_efficiency.csv") # csv for tasks - each pod is a task

pods = 1 # total number of tasks for job to run; should be number of rows in csv file
par = 1 # number of tasks that can be run in parallel - max. = # of pods

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