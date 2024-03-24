
# Getting started

- Clone the Git repository to your local machine with `git clone`.
- Navigate to the project directory using `cd`.
- Create a new branch with the Jira issue key in the name using `git branch <issue-key>-<branch-name>`. For example, `git branch SRP-1-DBAL`.
- Switch to the new branch with `git checkout <issue-key>-<branch-name>`.
- Develop your code on this branch and commit changes with `git commit`.
- Push the branch to the remote repository with `git push`.


Remember to replace <issue-key>-<branch-name> with your specific Jira issue key and desired branch name.


----

# Introduction

The purpose of this repository is to empirically prove the hypothesis that "regularization strength is inversely proportional to the AUBC or Loss". In order to achieve this, we will implement an active learning training model for all of our selected datasets and chosen baseline acquisition functions, as mentioned in point 2.

To conduct our experiments, we will explore various regularization methods, as mentioned in the research idea section. It is important to note that as we increase the number of active learning iterations, the size of the training data will also increase, leading to a change in distribution. Consequently, the hyper-parameters that work well in one active learning iteration may not perform well in subsequent iterations.

To address this concern, we will employ online hyper-parameter tuning at each active learning iteration in our implementation. This will involve using techniques such as grid search or random search to select the best regularization weights and methods. We will also track the changes in regularization hyperparameters throughout our experiments using online tuning. 

In the hypothesis phase, we will plot the trend of the regularization hyperparameters in each active learning iteration to analyze the results.

Additionally, each group member will initially implement their own Acquisition function, which will later be combined to form a comprehensive solution.


