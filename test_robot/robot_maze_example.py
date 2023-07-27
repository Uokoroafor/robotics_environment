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

    def load_maze(self, filename):
        with open(filename, "r") as file:
            for row, line in enumerate(file):
                for col, char in enumerate(line.strip()):
                    if char == " ":
                        self.maze[row][col] = 0

    def is_valid_cell(self, row, col):
        return MAZE_HEIGHT > row >= 0 == self.maze[row][col] and 0 <= col < MAZE_WIDTH

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
        self.maze.load_maze("maze.txt")

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
