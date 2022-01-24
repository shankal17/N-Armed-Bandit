# N-Armed-Bandit
This project follows along with **Chapter 2** of [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf), which presents a number of action selection policies for the *n*-armed bandit learning problem. The following policies are implemented:
- Epsilon-Greedy
- Softmax
- Upper-Confidence-Bound (UCB)

![comparison](/imgs/policy_comparison.PNG)


## Getting Started
Once you have the code, set up a virtual environment if you would like and install the necessary libraries by running the command below.
```bat
pip install -r /path/to/requirements.txt
```
From there, agents can be created and tested as seen in [these notebooks](https://github.com/shankal17/N-Armed-Bandit/tree/main/notebooks).

## Example Results
All policies act on a 10-armed testbed with true action values pulled from a zero mean and unit variance normal distribution.

![epsilon-greedy](/imgs/epsilon_greedy_policy.PNG)<br/>
*Visualization of epsilon-greedy policy performance*<br/><br/><br/>

![softmax](/imgs/softmax_policy.PNG)<br/>
*Visualization of softmax policy performance*<br/><br/><br/>

![UCB](/imgs/upper_confidence_bound_policy.PNG)<br/>
*Visualization of Upper-Confidence-Bound (UCB) policy performance*<br/><br/><br/>

![UCB](/imgs/optimistic_initial_values.PNG)<br/>
*Visualization of epsilon-greedy policy performance with optimistic initial value estimations*
