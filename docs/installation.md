<!-- ### Basic
```bash
pip install dreem-tracker
```  
-->
#### Clone the repository:
```bash
git clone https://github.com/talmolab/dreem && cd dreem
```
#### Set up in a new conda environment:
##### Linux/Windows:
###### GPU-accelerated (requires CUDA/nvidia gpu)
```bash
conda env create -f environment.yml && conda activate dreem
```
###### CPU:
```bash
conda env create -f environment_cpu.yml && conda activate dreem
```
#### OSX (Apple Silicon)
```bash
conda env create -f environment_osx-arm64.yml && conda activate dreem
```
### Uninstall
```
conda env remove -n dreem
```