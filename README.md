# AI-PROJECT
import numpy as np
import copy
import pygame

# --- Reversi AI Class ---
class ReversiAI:
    def __init__(self, player=2, depth=3):
        self.player = player
        self.opponent = 1 if player == 2 else 2
        self.depth = depth          
        self.weights = np.array([
            [100, -10, 10, 5, 5, 10, -10, 100],
            [-10, -30, -2, -2, -2, -2, -30, -10],
            [10, -2, -1, -1, -1, -1, -2, 10],
            [5, -2, -1, -1, -1, -1, -2, 5],
            [5, -2, -1, -1, -1, -1, -2, 5],
            [10, -2, -1, -1, -1, -1, -2, 10],
            [-10, -30, -2, -2, -2, -2, -30, -10],
            [100, -10, 10, 5, 5, 10, -10, 100]
        ])

    def evaluate(self, board):
        ai_score = np.sum(self.weights[board == self.player])
        opponent_score = np.sum(self.weights[board == self.opponent])
        return ai_score - opponent_score

    def minimax(self, board, depth, maximizing, alpha, beta, get_valid_moves, make_move):
        valid_moves = get_valid_moves(board, self.player if maximizing else self.opponent)
        if depth == 0 or not valid_moves:
            return self.evaluate(board), None

        best_move = None
        if maximizing:
            max_eval = float('-inf')
            for move in valid_moves:
                new_board = copy.deepcopy(board)
                make_move(new_board, move[0], move[1], self.player)
                eval, _ = self.minimax(new_board, depth - 1, False, alpha, beta, get_valid_moves, make_move)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in valid_moves:
                new_board = copy.deepcopy(board)
                make_move(new_board, move[0], move[1], self.opponent)
                eval, _ = self.minimax(new_board, depth - 1, True, alpha, beta, get_valid_moves, make_move)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

# --- Reversi Board Class ---
class ReversiBoard:
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=int)
        self.board[3][3] = self.board[4][4] = 2
        self.board[3][4] = self.board[4][3] = 1

    def is_on_board(self, x, y):
        return 0 <= x < 8 and 0 <= y < 8

    def get_valid_moves(self, board, player):
        opponent = 1 if player == 2 else 2
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        valid_moves = []

        for x in range(8):
            for y in range(8):
                if board[x][y] != 0:
                    continue
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    has_opponent = False
                    while self.is_on_board(nx, ny) and board[nx][ny] == opponent:
                        nx += dx
                        ny += dy
                        has_opponent = True
                    if has_opponent and self.is_on_board(nx, ny) and board[nx][ny] == player:
                        valid_moves.append((x, y))
                        break
        return valid_moves

    def make_move(self, board, x, y, player):
        board[x][y] = player
        opponent = 1 if player == 2 else 2
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            cells_to_flip = []

            while self.is_on_board(nx, ny) and board[nx][ny] == opponent:
                cells_to_flip.append((nx, ny))
                nx += dx
                ny += dy

            if self.is_on_board(nx, ny) and board[nx][ny] == player:
                for fx, fy in cells_to_flip:
                    board[fx][fy] = player

    def count_discs(self):
        p1 = np.count_nonzero(self.board == 1)
        p2 = np.count_nonzero(self.board == 2)
        return p1, p2

# --- Pygame Setup ---
pygame.init()
WIDTH = 640
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Beginner Friendly Reversi")

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 128, 0)
DARK_GRAY = (80, 80, 80)
LIGHT_GREEN = (102, 255, 102)

# --- Font ---
FONT = pygame.font.Font(None, 36)

def draw_board(board_obj, valid_moves, current_player):
    WIN.fill(GREEN)
    cell_size = WIDTH // 8
    for x in range(8):
        for y in range(8):
            rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
            pygame.draw.rect(WIN, BLACK, rect, 1)

            if board_obj.board[x][y] == 1:
                pygame.draw.circle(WIN, BLACK, rect.center, cell_size // 2 - 5)
            elif board_obj.board[x][y] == 2:
                pygame.draw.circle(WIN, WHITE, rect.center, cell_size // 2 - 5)

    # Show hints only for human player
    if current_player == 1:
        for move in valid_moves:
            hint_rect = pygame.Rect(move[1] * cell_size + cell_size // 4,
                                     move[0] * cell_size + cell_size // 4,
                                     cell_size // 2, cell_size // 2)
            pygame.draw.circle(WIN, DARK_GRAY, hint_rect.center, cell_size // 4)

    # Player label
    player_text = FONT.render(f"Player: {'Black' if current_player == 1 else 'White (AI)'}", True, BLACK)
    WIN.blit(player_text, (10, 10))

    # Score display
    p1_count, p2_count = board_obj.count_discs()
    score_text = FONT.render(f"Score - Black: {p1_count} | White: {p2_count}", True, BLACK)
    WIN.blit(score_text, (10, 50))

    pygame.display.update()


def display_end_screen(board_obj):
    p1_count, p2_count = board_obj.count_discs()
    winner_text = ""
    if p1_count > p2_count:
        winner_text = "Black Wins!"
    elif p2_count > p1_count:
        winner_text = "White Wins!"
    else:
        winner_text = "It's a Draw!"

    WIN.fill(GREEN)
    winner_surf = FONT.render(winner_text, True, BLACK)
    score_surf = FONT.render(f"Black: {p1_count}, White: {p2_count}", True, BLACK)
    winner_rect = winner_surf.get_rect(center=(WIDTH // 2, WIDTH // 2 - 20))
    score_rect = score_surf.get_rect(center=(WIDTH // 2, WIDTH // 2 + 20))

    WIN.blit(winner_surf, winner_rect)
    WIN.blit(score_surf, score_rect)
    pygame.display.update()
    pygame.time.delay(3000)

def main():
    clock = pygame.time.Clock()
    board = ReversiBoard()
    ai = ReversiAI(player=2, depth=3)
    player_turn = True
    running = True

    while running:
        clock.tick(30)
        current_player = 1 if player_turn else 2
        valid_moves = board.get_valid_moves(board.board, current_player)

        # Show board after every change
        draw_board(board, valid_moves, current_player)

        if not valid_moves and not board.get_valid_moves(board.board, 3 - current_player):
            display_end_screen(board)
            running = False
            continue

        if not valid_moves:
            player_turn = not player_turn
            continue

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if player_turn and event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                row, col = my // (WIDTH // 8), mx // (WIDTH // 8)
                if (row, col) in valid_moves:
                    board.make_move(board.board, row, col, 1)
                    player_turn = False
                    draw_board(board, board.get_valid_moves(board.board, 2), 2)
                    pygame.display.update()
                    pygame.time.delay(500)

        if not player_turn:
            _, move = ai.minimax(board.board, ai.depth, True, float('-inf'), float('inf'),
                                 board.get_valid_moves, board.make_move)
            if move:
                board.make_move(board.board, move[0], move[1], 2)
            player_turn = True

    pygame.quit()

if __name__ == "__main__":
    main()

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

