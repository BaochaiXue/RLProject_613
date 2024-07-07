Project Plan: Gym Environment Definition
Overview
The objective of this project is to define and implement a Gym environment for simulating and scheduling multiple deep learning (DL) tasks on edge devices. The environment will be configured using a JSON file that specifies various parameters, including the number of periodic DL tasks, the number of Poisson-distributed DL tasks, and the detailed configurations of each task. For each task, the environment will retrieve the model address, dataset type, and the number of model variants.

Steps to Define the Gym Environment
Reading the Configuration JSON File

Objective: Parse the JSON file to extract environment parameters.
Details:
The JSON file will include keys for the number of periodic and Poisson-distributed DL tasks.
Each task will have its own configuration, including the model address, dataset type, and the number of model variants.
For periodic tasks, the configuration will include the period, and the deadline of periodic tasks is equal to the period.
For Poisson-distributed tasks, the configuration will include the arrival rate and the deadline.
Environment Initialization

Objective: Initialize the Gym environment using the parsed configuration data.
Details:
Define the observation space and action space based on the task configurations.
Initialize any necessary internal states or variables, such as task queues, resource availability, and time.
Task Configuration

Objective: Define and configure each DL task according to the specifications in the JSON file.
Details:
Periodic DL tasks: Tasks that arrive at regular intervals with a specific period and deadline.
Poisson-distributed DL tasks: Tasks that arrive according to a Poisson process with a specified arrival rate and deadline.
Each task will also include parameters such as model address, dataset type, and the number of model variants.
Observation Space

Objective: Define the observation space for the Gym environment.
Details:
The observation space will represent the current state of the environment, including:
Current time
Status of each task (e.g., pending, running, completed)
Available resources
Model address, dataset type, and the number of model variants for each task
Any other relevant state information.
Action Space

Objective: Define the action space for the Gym environment.
Details:
The action space will represent the possible actions that can be taken by the scheduling agent.
Actions might include:
Allocating resources to tasks
Preempting or rescheduling tasks
Any other task management actions.
Reward Function

Objective: Define the reward function to guide the agent's learning process.
Details:
The reward function will be designed to encourage efficient and timely completion of tasks.
Factors influencing the reward might include:
Task completion time relative to deadlines
Resource utilization efficiency
Penalties for delayed or failed tasks.
Simulation Loop

Objective: Implement the main simulation loop of the Gym environment.
Details:
The loop will simulate the passage of time and the arrival of tasks.
At each time step, the environment will:
Update the state based on task arrivals and completions
Allow the agent to take actions
Calculate and provide rewards
Transition to the next state.
Execution of DL Tasks on GPUs

Objective: Execute DL tasks based on the RL agent's actions using GPU resources.
Details:
All DL tasks will be executed on GPUs.
Utilize two GPU streams for executing the tasks to parallelize the workload.
Testing and Validation

Objective: Test and validate the Gym environment to ensure it accurately simulates the task scheduling scenario.
Details:
Develop unit tests for individual components (e.g., task initialization, state updates).
Run end-to-end simulations to verify the environment's overall behavior.
Adjust configurations and parameters as needed based on test results.