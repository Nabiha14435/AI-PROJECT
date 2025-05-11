# AI-PROJECT
AI for Reversi (Othello) using an optimized Minimax algorithm with Alpha-Beta Pruning and adaptive heuristics.

In this project, the classic board game of Reversi has been developed using the Pygame framework. The game follows traditional rules but incorporates a smarter Artificial Intelligence (AI) opponent. The AI uses the Minimax algorithm with Alpha-Beta Pruning and a custom heuristic evaluation function that prioritizes strategic board positions like corners and edges. Additionally, the AI adapts over time based on the player's previous moves, providing an increasingly challenging experience.

Aim of the Project:
The goal of this project was to implement an intelligent AI for Reversi that can:

Simulate intelligent decision-making using the Minimax algorithm.

Optimize performance using Alpha-Beta Pruning.

Adapt to player behavior with heuristic evaluations.

Provide an engaging GUI for interactive gameplay using Pygame.

The focus was on making the AI stronger and more strategic, not just to win every time, but to emulate a human-like understanding of gameplay.

Minimax Algorithm with Alpha-Beta Pruning:
The AI assumes the opponent is playing optimally and generates a decision tree of possible game states. Each branch represents a player's move. Using Alpha-Beta Pruning, we reduce unnecessary branches, improving performance by avoiding evaluations of moves that won’t influence the final decision.

The tree is evaluated using a custom heuristic function, and scores are propagated back using:

Max value for the AI's move (trying to maximize its score).

Min value for the opponent's move (assuming they’ll minimize the AI’s advantage).

This approach allows the AI to plan several steps ahead efficiently.

Evaluation Function and Heuristic Design:
The evaluation function gives each board state a score using the following logic:

Terminal states (game over):

AI wins: +1000

AI loses: -1000

Ongoing states:

Corner pieces: +100

Edge pieces: +40

Center control: +20

Mobility (number of available moves): +10 per move

Opponent's mobility: -10 per move

The AI also learns from past games and adjusts its priorities to better counter the player's strategy.

Game Features:
🧠 Multiple Difficulty Modes: Easy, Medium, Hard.

🎮 Pygame GUI: Clean and interactive interface.

📈 Adaptive AI: Learns opponent patterns and adjusts moves.

⚡ Optimized AI Engine: Alpha-Beta Pruning improves move speed.

Game Mechanics:
Players take turns placing discs on an 8×8 board.

Legal moves flip opponent discs trapped in a line.

If no legal move is possible, the player’s turn is skipped.

The game ends when no legal moves remain.

The player with the most discs on the board wins.

Technologies Used:
Python

Pygame (GUI)

NumPy (Board logic)

Scikit-learn (for optional learning features)

Minimax with Alpha-Beta Pruning

Timeline Overview:
Week 1-2: Defined game rules and structure

Week 3-4: Implemented core AI logic and heuristic functions

Week 5-6: Built game board and move logic

Week 7: Integrated and optimized AI

Week 8: Final testing and documentation

Screenshots:
<img src="images/reversi_board.png" width="45%"> <img src="images/reversi_ai_move.png" width="45%">

References:
Online Reversi Strategy Guides

Game AI Programming Techniques by Mat Buckland

Research on Minimax and Alpha-Beta Pruning

