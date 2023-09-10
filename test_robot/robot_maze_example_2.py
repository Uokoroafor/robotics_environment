import arcade
import random

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Maze Robot"

CELL_SIZE = 40
MAZE_WIDTH = SCREEN_WIDTH // CELL_SIZE
MAZE_HEIGHT = SCREEN_HEIGHT // CELL_SIZE


class Maze:
    def __init__(self):
        self.maze = [[1] * MAZE_WIDTH for _ in range(MAZE_HEIGHT)]

    def generate_maze(self):
        stack = [(0, 0)]
        visited = {(0, 0)}

        while stack:
            current_cell = stack[-1]
            neighbors = self.get_unvisited_neighbors(current_cell, visited)

            if neighbors:
                next_cell = random.choice(neighbors)
                self.remove_wall(current_cell, next_cell)
                visited.add(next_cell)
                stack.append(next_cell)
            else:
                stack.pop()

    def get_unvisited_neighbors(self, cell, visited):
        row, col = cell
        neighbors = []

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for dx, dy in directions:
            next_row, next_col = row + dx, col + dy
            if (
                self.is_valid_cell(next_row, next_col)
                and (next_row, next_col) not in visited
            ):
                neighbors.append((next_row, next_col))

        return neighbors

    def is_valid_cell(self, row, col):
        return 0 <= row < MAZE_HEIGHT and 0 <= col < MAZE_WIDTH

    def remove_wall(self, current_cell, next_cell):
        current_row, current_col = current_cell
        next_row, next_col = next_cell

        if current_row < next_row:
            self.maze[current_row + 1][current_col] = 0
        elif current_row > next_row:
            self.maze[current_row - 1][current_col] = 0
        elif current_col < next_col:
            self.maze[current_row][current_col + 1] = 0
        elif current_col > next_col:
            self.maze[current_row][current_col - 1] = 0

    def draw(self):
        for row in range(MAZE_HEIGHT):
            for col in range(MAZE_WIDTH):
                if self.maze[row][col] == 1:
                    x = col * CELL_SIZE
                    y = row * CELL_SIZE
                    arcade.draw_rectangle_filled(
                        x, y, CELL_SIZE, CELL_SIZE, arcade.color.BLACK
                    )


class Robot(arcade.Sprite):
    def __init__(self, x, y):
        super().__init__("robot.png", 0.5)

        self.center_x = x
        self.center_y = y

    def move_up(self):
        self.center_y += CELL_SIZE

    def move_down(self):
        self.center_y -= CELL_SIZE

    def move_left(self):
        self.center_x -= CELL_SIZE

    def move_right(self):
        self.center_x += CELL_SIZE


class MazeRobot(arcade.Window):
    def __init__(self, width, height, title):
        super().__init__(width, height, title)

        arcade.set_background_color(arcade.color.CREAM)

        self.maze = Maze()
        self.maze.generate_maze()

        self.robot = Robot(CELL_SIZE // 2, CELL_SIZE // 2)
        # Place goal in a random cell
        self.goal_x = random.randint(0, MAZE_WIDTH - 1) * CELL_SIZE + CELL_SIZE // 2
        self.goal_y = random.randint(0, MAZE_HEIGHT - 1) * CELL_SIZE + CELL_SIZE // 2
        # self.goal_x = (MAZE_WIDTH - 1) * CELL_SIZE + CELL_SIZE // 2
        # self.goal_y = (MAZE_HEIGHT - 1) * CELL_SIZE + CELL_SIZE // 2

    def on_draw(self):
        arcade.start_render()

        self.maze.draw()
        self.robot.draw()

        arcade.draw_rectangle_filled(
            self.goal_x, self.goal_y, CELL_SIZE, CELL_SIZE, arcade.color.GREEN
        )

    def update(self, delta_time):
        pass

    def on_key_press(self, key, modifiers):
        if key == arcade.key.UP:
            if (
                self.robot.center_y < SCREEN_HEIGHT - CELL_SIZE
                and self.maze.maze[self.robot.center_y // CELL_SIZE + 1][
                    self.robot.center_x // CELL_SIZE
                ]
                == 0
            ):
                self.robot.move_up()
        elif key == arcade.key.DOWN:
            if (
                self.robot.center_y > CELL_SIZE
                and self.maze.maze[self.robot.center_y // CELL_SIZE - 1][
                    self.robot.center_x // CELL_SIZE
                ]
                == 0
            ):
                self.robot.move_down()
        elif key == arcade.key.LEFT:
            if (
                self.robot.center_x > CELL_SIZE
                and self.maze.maze[self.robot.center_y // CELL_SIZE][
                    self.robot.center_x // CELL_SIZE - 1
                ]
                == 0
            ):
                self.robot.move_left()
        elif key == arcade.key.RIGHT:
            if (
                self.robot.center_x < SCREEN_WIDTH - CELL_SIZE
                and self.maze.maze[self.robot.center_y // CELL_SIZE][
                    self.robot.center_x // CELL_SIZE + 1
                ]
                == 0
            ):
                self.robot.move_right()


def main():
    _ = MazeRobot(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    arcade.run()


if __name__ == "__main__":
    main()
