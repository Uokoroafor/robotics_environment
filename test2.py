import arcade
import math
import random

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

class Pendulum(arcade.Window):
    def __init__(self, width, height):
        super().__init__(width, height, "Swinging Pendulum")
        self.pendulum_length = 200
        self.pendulum_mass = 10
        # Want a random angle to start with
        self.angle = random.uniform(-math.pi/2, math.pi/2)
        # Want a random angular velocity to start with
        self.init_angular_velocity = random.uniform(-0.05, 0.05)

        self.pivot = None
        self.pendulum_body = None
        self.saved = False


    def setup(self):
        arcade.set_background_color(arcade.color.AMAZON)

        # Set up the pivot point
        self.pivot = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)

        # Create the pendulum
        x = self.pivot[0]
        y = self.pivot[1] - self.pendulum_length
        self.pendulum_body = (x, y)

    def on_draw(self):
        arcade.start_render()

        # Draw the pendulum line
        arcade.draw_line(self.pivot[0], self.pivot[1], self.pendulum_body[0], self.pendulum_body[1], arcade.color.BLACK,2)

        # Draw the pendulum bob
        arcade.draw_circle_filled(self.pendulum_body[0], self.pendulum_body[1], 10, arcade.color.RED)
        # save initial render as image
        if not self.saved:
            # arcade.finish_render()
            arcade.get_image().save('test2.png')
            self.saved = True




    def update(self, delta_time):
        # Update the pendulum angle
        angular_velocity = self.init_angular_velocity  # Adjust this value to change the swing speed
        self.angle += angular_velocity

        # Update the position of the pendulum bob based on the angle
        self.pendulum_body = (
            self.pivot[0] + math.sin(self.angle) * self.pendulum_length,
            self.pivot[1] - math.cos(self.angle) * self.pendulum_length
        )
        # Create stop condition
        if self.angle > math.pi*0.8 or self.angle < -math.pi*0.8:
            print('The pendulum has completed a full swing')
            arcade.close_window()

    def on_key_press(self, key, modifiers):
        if key == arcade.key.ESCAPE:
            arcade.close_window()

def main():
    game = Pendulum(SCREEN_WIDTH, SCREEN_HEIGHT)
    game.setup()
    arcade.run()

if __name__ == "__main__":
    main()
