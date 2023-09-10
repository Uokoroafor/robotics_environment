import arcade
import random

# Maze size
WIDTH = 21
HEIGHT = 21
CELL_SIZE = 30
SCREEN_WIDTH = WIDTH * CELL_SIZE
SCREEN_HEIGHT = HEIGHT * CELL_SIZE
SCREEN_TITLE = "Maze Visualization"

# Maze representation
maze = [[1] * WIDTH for _ in range(HEIGHT)]

# Directions
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)


# Check if a cell is within the maze bounds
def is_valid_cell(row, col):
    return row >= 0 and row < HEIGHT and col >= 0 and col < WIDTH


# Generate the maze using Recursive Backtracking algorithm
def generate_maze(row, col):
    maze[row][col] = 0  # Mark the current cell as visited

    directions = [UP, DOWN, LEFT, RIGHT]
    random.shuffle(directions)

    for direction in directions:
        dx, dy = direction
        next_row, next_col = row + 2 * dx, col + 2 * dy

        if is_valid_cell(next_row, next_col) and maze[next_row][next_col] == 1:
            maze[row + dx][col + dy] = 0  # Carve a path
            generate_maze(next_row, next_col)


class MazeVisualization(arcade.Window):
    def __init__(self, width, height, title):
        super().__init__(width, height, title)

        arcade.set_background_color(arcade.color.WHITE)

    def setup(self):
        generate_maze(0, 0)

    def on_draw(self):
        arcade.start_render()

        for row in range(HEIGHT):
            for col in range(WIDTH):
                if maze[row][col] == 1:
                    x = col * CELL_SIZE
                    y = row * CELL_SIZE
                    arcade.draw_rectangle_filled(
                        x, y, CELL_SIZE, CELL_SIZE, arcade.color.BLACK
                    )


def main():
    visualization = MazeVisualization(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    visualization.setup()
    arcade.run()


if __name__ == "__main__":
    main()
