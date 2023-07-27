import arcade
import random

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Robot Maze"

CELL_SIZE = 40
MAZE_WIDTH = SCREEN_WIDTH // CELL_SIZE
MAZE_HEIGHT = SCREEN_HEIGHT // CELL_SIZE

class Maze:
    def __init__(self):
        self.maze = [[1] * MAZE_WIDTH for _ in range(MAZE_HEIGHT)]

    def generate_maze(self):
        stack = [(0, 0)]
        visited = [[False] * MAZE_WIDTH for _ in range(MAZE_HEIGHT)]
        visited[0][0] = True

        while stack:
            current_cell = stack[-1]
            row, col = current_cell

            neighbors = self.get_unvisited_neighbors(row, col, visited)
            if neighbors:
                next_cell = random.choice(neighbors)
                next_row, next_col = next_cell

                self.remove_wall(current_cell, next_cell)

                visited[next_row][next_col] = True
                stack.append(next_cell)
            else:
                stack.pop()

    def get_unvisited_neighbors(self, row, col, visited):
        neighbors = []

        if row > 1 and not visited[row - 2][col]:
            neighbors.append((row - 2, col))
        if col < MAZE_WIDTH - 2 and not visited[row][col + 2]:
            neighbors.append((row, col + 2))
        if row < MAZE_HEIGHT - 2 and not visited[row + 2][col]:
            neighbors.append((row + 2, col))
        if col > 1 and not visited[row][col - 2]:
            neighbors.append((row, col - 2))

        return neighbors

    def remove_wall(self, cell1, cell2):
        row1, col1 = cell1
        row2, col2 = cell2

        self.maze[(row1 + row2) // 2][(col1 + col2) // 2] = 0

    def is_valid_cell(self, row, col):
        print(row, col)
        row = int(row)
        col = int(col)

        return row >= 0 and row < MAZE_HEIGHT and col >= 0 and col < MAZE_WIDTH and self.maze[row][col] == 0

    def draw(self):
        for row in range(MAZE_HEIGHT):
            for col in range(MAZE_WIDTH):
                if self.maze[row][col] == 1:
                    x = col * CELL_SIZE
                    y = row * CELL_SIZE
                    arcade.draw_rectangle_filled(x, y, CELL_SIZE, CELL_SIZE, arcade.color.BLACK)

class Robot(arcade.Sprite):
    def __init__(self, filename, scale, maze):
        super().__init__(filename, scale)

        self.center_x = CELL_SIZE // 2
        self.center_y = CELL_SIZE // 2

        self.maze = maze

    def update(self):
        new_x = self.center_x + self.change_x
        new_y = self.center_y + self.change_y

        if self.maze.is_valid_cell(new_y // CELL_SIZE, new_x // CELL_SIZE):
            self.center_x = new_x
            self.center_y = new_y

class RobotMaze(arcade.Window):
    def __init__(self, width, height, title):
        super().__init__(width, height, title)

        arcade.set_background_color(arcade.color.WHITE)

        self.maze = Maze()
        self.maze.generate_maze()

        self.robot = Robot("robot.png", 0.5, self.maze)
        self.robot.center_x = CELL_SIZE // 2
        self.robot.center_y = CELL_SIZE // 2

    def on_draw(self):
        arcade.start_render()

        self.maze.draw()
        self.robot.draw()

    def update(self, delta_time):
        self.robot.update()

    def on_key_press(self, key, modifiers):
        if key == arcade.key.UP:
            self.robot.change_x = 0
            self.robot.change_y = CELL_SIZE
        elif key == arcade.key.DOWN:
            self.robot.change_x = 0
            self.robot.change_y = -CELL_SIZE
        elif key == arcade.key.LEFT:
            self.robot.change_x = -CELL_SIZE
            self.robot.change_y = 0
        elif key == arcade.key.RIGHT:
            self.robot.change_x = CELL_SIZE
            self.robot.change_y = 0

    def on_key_release(self, key, modifiers):
        if key == arcade.key.UP or key == arcade.key.DOWN:
            self.robot.change_y = 0
        elif key == arcade.key.LEFT or key == arcade.key.RIGHT:
            self.robot.change_x = 0

def main():
    game = RobotMaze(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    arcade.run()

if __name__ == "__main__":
    main()
