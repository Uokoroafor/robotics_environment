import arcade

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Text Instruction Simulation"


class Robot(arcade.Sprite):
    def __init__(self, filename, scale):
        super().__init__(filename, scale)

        self.center_x = SCREEN_WIDTH // 2
        self.center_y = SCREEN_HEIGHT // 2

        self.change_x = 0
        self.change_y = 0

    def update(self):
        self.center_x += self.change_x
        self.center_y += self.change_y


class InstructionSimulation(arcade.Window):
    def __init__(self, width, height, title):
        super().__init__(width, height, title)

        arcade.set_background_color(arcade.color.WHITE)

        self.robot = Robot("robot.png", 0.5)

    def on_draw(self):
        arcade.start_render()
        self.robot.draw()

    def update(self, delta_time):
        self.robot.update()

    def on_key_press(self, key, modifiers):
        if key == arcade.key.UP:
            self.robot.change_y = 1
        elif key == arcade.key.DOWN:
            self.robot.change_y = -1
        elif key == arcade.key.LEFT:
            self.robot.change_x = -1
        elif key == arcade.key.RIGHT:
            self.robot.change_x = 1

    def on_key_release(self, key, modifiers):
        if key == arcade.key.UP or key == arcade.key.DOWN:
            self.robot.change_y = 0
        elif key == arcade.key.LEFT or key == arcade.key.RIGHT:
            self.robot.change_x = 0


def main():
    _ = InstructionSimulation(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    arcade.run()


if __name__ == "__main__":
    main()
