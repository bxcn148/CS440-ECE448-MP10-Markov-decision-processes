Download link :https://programming.engineering/product/cs440-ece448-mp10-markov-decision-processes/

# CS440-ECE448-MP10-Markov-decision-processes
CS440/ECE448 MP10: Markov decision processes
The first thing you need to do is to download this file: mp10.zip. It has the following

content:

submitted.py



Your homework. Edit, and then submit to Gradescope.

mp10_notebook.ipynb



This is a Jupyter notebook to help you debug. You can

completely ignore it if you want, although you might find that it gives you useful

instructions.

grade.py



Once your homework seems to be working, you can test it by typing

python grade.py tests/tests_visible.py

, which will run the tests in .

tests/test_visible.py



This file contains about half of the unit tests that

Gradescope will run in order to grade your homework. If you can get a perfect score on these tests, then you should also get a perfect score on the additional hidden tests that Gradescope uses.

solution.json



This file contains the solutions for the visible test cases, in JSON

format. If the instructions are confusing you, please look at this file, to see if it can help to clear up your confusion.

models



This directory contains two MDP models. Especially,

models/model_small.json

is exactly the same as the one presented in the

slides.

utils.py



This is an auxiliary program that you can use to load the model and

visualize it.

Please note that there is no extra packages that you should be using except for NumPy.

(Using exsiting MDP libraries would result in score 0!)

mp10_notebook.ipynb

This file ( ) will walk you through the whole MP, giving you instructions and debugging tips as you go.

Table of Contents

The MDP environment

Value iteration

Grade Your Homework

The MDP environment

4/21/24, 1:03 PM mp10_notebook

In this MP, you will implement the value iteration algorithm introduced in the class. The MDPs you will work on are similar to the grid world example mentioned in the class, but

with state-dependent transition and reward model.

Loading the MDP model

utils.py

Helper functions are provided in . Two predefined MDP models are given in

models models/small.json

. Please note that defines exactly the same MDP model

presented in the lecture, and you can use the intermediate results in the slides to debug

load_MDP(filename)


your implementation. With function , you can load a MDP model

as follows.


4/21/24, 1:03 PM mp10_notebook

of each state when the policy is kept constant throughout the evaluation process. Unlike

model.FP

the dynamic policy iteration where the policy might evolve over time,

remains unchanged, making it possible to analyze the performance of a specific policy.

Visualization

We also provide a helper function for visualizing the environment, and the utility function. To use it, please run the following. In the figure, “x” marks a cell that is occupied by the wall. “+1” and “-1” mark the terminal states and their rewards.


4/21/24, 1:03 PM mp10_notebook


Coordinate system

Please be aware of the coordinate system we will use in this MP. In the above

visualization, the cell at the upper-left corner is (0, 0), the upper-right is (0, 3), and bottom-left is (2, 0). Moving up means moving from (r, c) to (r − 1, c), moving right means from (r, c) to (r, c + 1), and so on.

Value iteration

As stated in the lecture, the utility of a state s is the best possible expected sum of discounted rewards and denoted by U (s). With value iteration, we can compute this function U. The algorithm proceeds as follows.

We start with iteration i = 0 and simply initialize Ui(s) = 0 for all s. Then at each iteration, we update U as follows

Ui+1(s) = R(s) + γ max ∑ P(s′|s, a)Ui(s′).

a s′

We keep doing this until convergence, i.e., when |Ui+1(s) − Ui(s)| < ϵ for all s, where

> 0 is a constant.

In order to implement the algorithm, you need to complete the following functions in

submitted.py

.

4/21/24, 1:03 PM mp10_notebook

Computing the transition matrix P

First, notice that the transition matrix P(s′|s, a) will be called many times, and it will not change during the value iteration. Thus, it makes sense to precompute it before doing the value iteration. To this end, you need to complete the function

compute_transition() model

. This function takes in the MDP model and

computes the transition “matrix”, which is actually an

M × N × 4 × M × N numpy

P

array

. In this function, you need to conside

r each state (r, c) and each action

P[r, c, a, r’, c’]

a ∈ {0 (left), 1 (up), 2 (right), 3 (down)}.

should be the

(r, c)

(r’, c’)

probability that the agent will move from cell

to

if it takes action


a (r, c) P[r, c, :, :, :]


. Especially, if is a terminal state, you can simply set

= 0

i.e., the probability that the agent move from a terminal state to any state

(including itself) is 0, since once the agent reaches a terminal state, the game is over.

P


You may notice that the transition matrix is very sparse, i.e., most of its elements are zeros. Better data structre such as sparse matrices can be used to improve the efficiency. But in this MP, we simply use a regular numpy array.

In MP10, you need to account for transition probabilities from wall states (current state is wall), which may have non-zero values, unlike typical scenarios. Logically, walls should not have a transition probability. However, for terminal states, all transition probabilities from them should be zero.


In [5]: import submitted, importlib

importlib.reload(submitted)

help(submitted.compute_transition)

Help on function compute_transition in module submitted:

compute_transition(model)

Parameters:

model – the MDP model returned by load_MDP()

Output:

P – An M x N x 4 x M x N numpy array. P[r, c, a, r’, c’] is the probabil ity that the agent will move from cell (r, c) to (r’, c’) if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).

‘models/model_small.json’

If you loaded the MDP in the previous section, you

can check some cells in the computed transition matrix to see if it is correct. For

P[1, 0, 2, :, :]

example, in the following, we check . Recall that this should the probability distribution of the next state if the agent takes the action 2 (right) at cell (1, 0). Please also keep in mind that cell (1, 1) is occupied by the wall. So, with

probability 0.1 the agent will move up to (0, 0); with probability 0.1 the agent will move down to (2, 0); with probability 0.8, it will move as intended (right) but will cause a collision to the wall, and thus the agent will stay at (1, 0) with probability 0.8.

4/21/24, 1:03 PM mp10_notebook


In [6]: P = submitted.compute_transition(model)

print(P[1, 0, 2, :, :])

model.visualize()

[[0.1 0. 0. 0. ]

[0.8 0. 0. 0. ]

[0.1 0. 0. 0. ]]


Updating the utility function

Then, you need to complete the function

compute_utility

, which takes in the

current utility function

U_current

(correspon

ding to the Ui in the above equation) and

U_next

computes the updated utility function

(corresponding to the Ui+1 in the above

equation). This function should implement the update rule (the equation) in the value iteration algorithm.


In [7]: importlib.reload(submitted)

help(submitted.compute_utility)

Help on function compute_utility in module submitted:

compute_utility(model, U_current, P)

Parameters:

model – The MDP model returned by load_MDP()

U_current – The current utility function, which is an M x N array

P – The precomputed transition matrix returned by compute_transition()

Output:

U_next – The updated utility function, which is an M x N array

4/21/24, 1:03 PM mp10_notebook

P U_current


Since we have represented the transition and utility as numpy arrays.

The best way to implement this function is to use vectorization. That is, we can rewrite

the update rule as some matrix operations and then use numpy’s builtin functions to

compute them. For example, the summation in the equation is actually an inner product

dot


of P and Ui. Using numpy’s function to compute this inner product is much faster than implementing it as a for loop. However, using vectorization is totally optional for

you. The efficiency of your program will not contribute to your score. You will not get any extra credit even if you indeed use vectorization. So feel free to use for loop since it is

much easier to implement.

Putting them together

value_iterate

Now, you are ready to complete the function, which should first

compute the

P

but calling

compute_transition

and then keeps calling

compute_utility

until convergence. Please keep in mind that the convergence

criterion is |Ui+1

(s) − Ui(s)| < ϵ for all s. In this MP

, please use ϵ = 10

−3. In

submitted.py

, you can find a predefined variable

epsilon = 1e-3

. Also, please

stop the program after a specifc number of iteration even if it has not converged. 100 iterations should be sufficient for all the tests in this MP.


In [8]: importlib.reload(submitted)

help(submitted.value_iterate)

Help on function value_iterate in module submitted:

value_iterate(model)

Parameters:

model – The MDP model returned by load_MDP()

Output:

U – The utility function, which is an M x N array

For the purpose of debugging, you can visualize the utility function at each iteration

model.visualize(U_current)

using the provided function to see how the utility is being updated. You can also compare your utility function to the ground truth presented

in the slides. For example, the following code visualize the computed utility. You can

compare it to the one in the slides.


4/21/24, 1:03 PM mp10_notebook


Grade your homework

If you’ve reached this point, and all of the above sections work, then you’re ready to try grading your homework! Before you submit it to Gradescope, try grading it on your own machine. This will run some visible test cases (which you can read in

tests/test_visible.py

), and compare the results to the solutions (which you can

solution.json

read in ).

The exclamation point (!) tells python to run the following as a shell command. Obviously you don’t need to run the code this way — this usage is here just to remind you that you can also, if you wish, run this command in a terminal window.


In [10]: !python grade.py

….

———————————————————————-

Ran 4 tests in 0.003s

OK

If you got any ‘E’ marks, it means that your code generated some runtime errors, and you need to debug those.

If you got any ‘F’ marks, it means that your code ran without errors, but that it generated

solutions.json

results that are different from the solutions in . Try debugging those differences.

4/21/24, 1:03 PM mp10_notebook

If neither of those things happened, and your result was a series of dots, then your code works perfectly.

If you’re not sure, you can try running grade.py with the -j option. This will produce a JSON results file, in which the best score you can get is 60.

submitted.py

Now you should try uploading to Gradescope.

Gradescope will run the same visible tests that you just ran on your own machine, plus some additional hidden tests. It’s possible that your code passes all the visible tests, but fails the hidden tests. If that happens, then it probably means that you hard-coded a number into your function definition, instead of using the input parameter that you were supposed to use. Debug by running your function with a variety of different input parameters, and see if you can get it to respond correctly in all cases.

Once your code works perfectly on Gradescope, with no errors, then you are done with the MP. Congratulations!

Extra Credit: Policy Evaluation Implementation

In this section, we extend our MDP solver to include policy evaluation, an essential

concept from the lectures. Policy evaluation is a process where we compute the utility of

following a given policy.

Fixed Policy Transition Matrix

We assume a fixed policy represented by a four-dimensional transition matrix

model.FP[r, c, r’, c’]

This matrix provides the probability of transitioning from

each state to every other state under the fixed policy. The dimensions correspond to the

(r, c) (r’, c’)


current state and the next state , respectively.

In the previous section, we use value iteration to dynamically choose actions that

maximize expected utility, aiming to find the optimal policy. This method continuously

updates the policy based on the calculated utilities of states. However, in this section the

focus shifts to evaluating the utility of states under a fixed policy, which remains

constant throughout the process. This evaluation does not seek to optimize the policy

but rather to assess the effectiveness of a given, unchanging strategy.

Implementation Overview

policy_evaluation

The function iteratively computes the expected utility of each

state under the given policy until the utility function converges. The convergence is

epsilon

determined by a small threshold .

4/21/24, 1:03 PM mp10_notebook

The utility of a state under the policy is computed using the following equation as shown in lecture:

ui(s) = r(s) + γ ∑ P(s′|s, πi(s))ui(s′)

s′

where:

ui(s) is the utility of state s at iteration i.



r(s) is the immediate reward received when entering state s.



is the discount factor that balances the importance of immediate and future

rewards.

P(s′|s, πi(s)) is the transition probability of moving from state s to state s′ under policy πi.



The summation is over all possible next states s′.

(r, c)


During each iteration, for every state , we calculate the expected utility as the

sum of the utilities of all possible next states

(r’, c’)

, weighted by the transition

probabilities from the current state

to the next state

. We then

(r, c)

(r’, c’)

update the utility of the current state with the immediate reward plus the discounted expected utility.

It’s important to note that the provided policy (encoded in the transition matrix

model.FP

may not necessarily be the optimal policy. It is fixed for the purpose of evaluating how good the policy is in terms of the utility it yields.

It may take more than 100 iterations to converge.


In [11]: importlib.reload(submitted)

help(submitted.policy_evaluation)

Help on function policy_evaluation in module submitted:

policy_evaluation(model)

Parameters:

model – The MDP model returned by load_MDP();

4/21/24, 1:03 PM mp10_notebook


In [13]: !python grade_extra.py

..

———————————————————————-

Ran 2 tests in 0.034s

OK

submitted.py

Now you should try uploading to Gradescope.

Gradescope will run the same visible tests that you just ran on your own machine, plus some additional hidden tests. It’s possible that your code passes all the visible tests, but fails the hidden tests. If that happens, then it probably means that you hard-coded a number into your function definition, instead of using the input parameter that you were supposed to use. Debug by running your function with a variety of different input parameters, and see if you can get it to respond correctly in all cases.

https://courses.grainger.illinois.edu/ece448 mp/mp10_notebook.html 12/12
