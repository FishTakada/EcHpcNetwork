# Entorinal-Hippoocampal network model (Takada and Tateno, 2022)
This is a repository of programs that reproduce the simulation results of the Takada and Tateno paper (2022). The repository includes cusnn, a library for fast computation of SNN using CUDA.

# Requirements
- CUDA 11.0
- gym 0.17.2
- matplotlib 3.2.2
- numpy 1.18.5
- pandas 1.4.0
- progressbar2 3.51.3
- pycuda 2019.1.2
- scipy 1.5.0
- seaborn 0.10.1

# Usage
1. Clone the repository
    ```commandline
    $ git clone https://github.com/FishTakada/EcHpcNetwork.git
    ```
1. Install the requirements
    
1. Run the simulation

    Use the following command to run simulation in a linear track environment.
    ```commandline
    $ python3 linear_track_simulation.py "/output_directory/"
    ```
    Use the following command to run simulation in a water maze environment.
    ```commandline
    $ python3 water_maze_simulation.py "/output_directory/"
    ```