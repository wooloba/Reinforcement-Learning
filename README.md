# Reinforcement-Learning

This repo shows the track of me learning reinforcement learning. I learn by reading the book of Reinforcement Learning written by Richard Sutton. To enforce my knowledge, I solved some problems in the book and implement algorithms described in the book to solve small games/puzzles.

The content includes:
1. Gambler's problem implemented by on-policy Monte Carlo Control with exploring starts for action values.
    A Gambler's problem is described in the book as such:  
        A gambler has the opportunity to make bets on the outcomes of a sequence of coin ﬂips. If the coin comes up heads, he wins as many dollars as he has staked on that ﬂip; if it is tails, he loses his stake. The game ends when the gambler wins by reaching his goal of $100, or loses by running out of money. On each ﬂip, the gambler must decide what portion of his capital to stake, in integer numbers of dollars.

2. Windy Grid world is implemented by Sarsa Algorithm which is also knwon as TD(temporal difference learning)

    a Windy world is described in the book as such:
        For a standard gridworld, with start and goal states. There is a crosswind upward through the middle of the grid. The actions are the standard four — up, down, right, and left(in my case, I used a switch that can add four more actions in the game which are up-left, up-right, down-left, down-right) but in the middle region the resultant next states are shifted upward by a “wind,” the strength of which varies from column to column. The strength of the wind is given below each column, in number of cells shifted upward. For example, if you are one cell to the right of the goal, then the action left takes you to the cell just above the goal. Let us treat this as an undiscounted episodic task, with constant rewards of '1 until the goal state is reached.

3. Maze is a modified version of Windy Grid World which looks like the following image and with no wind. The maze is solved by Dyna-Q agent. The task for agent is to travel from S to G as quickly as possible.![image-20180908224340457](/var/folders/yx/5ngc1rg53qn0zj53trvqhfzr0000gn/T/abnerworks.Typora/image-20180908224340457.png)

4. Random walk is implemented by three different agents based on TD(0). They are Tabular agent; tile coding agent; state aggregation agent.

Random walk is described in the book as such:
    All episodes start in the center state, C, and proceed either left or right by one state on each step, with equal probability. Episodes terminate either on the extreme left or the extreme right. When an episode terminates on the right, a reward of +1 occurs; all other rewards are zero.


Reinforcement learning is my entrance to ML/DL. Some concepts in Reinforcement Learning share with ML/DL. The interface of User-Agent make me feel comfortable to learn the idea of RL beside complex Math. The algorithms in the book is not very hard to understand and to implement. Therefore after reading this book, I deside to explore more in AI area.
