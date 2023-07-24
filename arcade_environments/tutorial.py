import math
import random
import arcade

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Pendulum"
GRAVITY = (0, -10)
PIVOTMASS = 100000000
PIVOT_MOI = 10000000000
PIVOT_FRICTION = 0.5

PENDULUM_MASS = 1
PENDULUM_MOI = 1
PENDULUM_FRICTION = 0.5
PENDULUM_LENGTH = 100
PENDULUM_WIDTH = 10
PENDULUM_HEIGHT = 10
PENDULUM_DAMPING = 0.1
PENDULUM_MAX_ANGULAR_VELOCITY = 10
PENDULUM_MAX_ANGLE = math.pi / 2


class MyPendulum(arcade.Window):
    """
    Main application class.

    Want to create an environment with a pendulum. It is initialised with a random angle and angular velocity.
    THen continues until it reaches a terminal state (angle = 0, angular velocity = 0)
    """

    def __init__(self, width, height, title):
        super().__init__(width, height, title)

        arcade.set_background_color(arcade.color.AMAZON)

        # Create the pendulum sprite
        self.pendulum_sprite = arcade.SpriteCircle(radius=10, color=arcade.color.WHITE)
        # Want a random position and velocity for the pendulum
        pos_x = random.uniform(SCREEN_WIDTH / 2 - 100, SCREEN_WIDTH / 2 + 100)
        pos_y = random.uniform(SCREEN_HEIGHT / 2 - 100, SCREEN_HEIGHT / 2 + 100)
        angle = random.uniform(0, 2 * math.pi)
        angular_velocity = random.uniform(-PENDULUM_MAX_ANGULAR_VELOCITY, PENDULUM_MAX_ANGULAR_VELOCITY)

        self.pendulum_sprite.set_position(pos_x, pos_y)
        self.pendulum_sprite.angle = angle
        self.pendulum_sprite.change_angle = angular_velocity

        # Create the pivot sprite
        self.pivot_sprite = arcade.SpriteCircle(radius=5, color=arcade.color.RED)
        self.pivot_sprite.set_position(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

        # Create the line sprite
        # Line has to originate from the pivot sprite and end at the pendulum sprite
        # self.line_sprite = arcade.SpriteSolidColor(PENDULUM_LENGTH,1, arcade.color.WHITE)
        # # Set one end of the line to the pivot sprite
        # self.line_sprite.center_x = (self.pivot_sprite.center_x+self.pendulum_sprite.center_x)/2
        # self.line_sprite.center_y = (self.pivot_sprite.center_y+self.pendulum_sprite.center_y)/2
        # # Set the other end of the line to the pendulum sprite
        # self.line_sprite.angle = self.pendulum_sprite.angle
        # self.line_sprite.change_angle = self.pendulum_sprite.change_angle

        # Create the physics engine
        self.physics_engine = arcade.PymunkPhysicsEngine(gravity=GRAVITY)

        # Add the sprites to the physics engine
        self.physics_engine.add_sprite(self.pendulum_sprite, mass=PENDULUM_MASS, moment=PENDULUM_MOI,
                                       friction=PENDULUM_FRICTION, damping=PENDULUM_DAMPING)
        # self.physics_engine.add_sprite(self.pivot_sprite, mass=PIVOTMASS, moment=PIVOT_MOI, friction=PIVOT_FRICTION)
        # self.physics_engine.add_sprite(self.line_sprite, mass=0, moment=0, friction=0)
        # Pivot sprite is static, so we don't need to add it to the physics engine

    def setup(self):
        """ Set up the game variables. Call to re-start the game. """
        # Create your sprites and sprite lists here
        pass

    def on_draw(self):
        """
        Render the screen.
        """

        # This command should happen before we start drawing. It will clear
        # the screen to the background color, and erase what we drew last frame.
        arcade.start_render()
        self.clear()
        # Draw the sprites
        self.pendulum_sprite.draw()
        self.pivot_sprite.draw()
        arcade.draw_line(self.pivot_sprite.position[0], self.pivot_sprite.position[1],
                         self.pendulum_sprite.position[0], self.pendulum_sprite.position[1],
                         arcade.color.WHITE, 1)

    def on_update(self, delta_time):
        """
        All the logic to move, and the game logic goes here.
        Normally, you'll call update() on the sprite lists that
        need it.
        """

        # Update the physics engine
        self.physics_engine.step()

        # # Update the line sprite
        # self.line_sprite.angle = self.pendulum_sprite.angle
        # self.line_sprite.center_x = self.pivot_sprite.center_x
        # self.line_sprite.center_y = self.pivot_sprite.center_y

        # Update the pendulum sprite
        self.pendulum_sprite.center_x = self.pivot_sprite.center_x + PENDULUM_LENGTH * math.sin(
            self.pendulum_sprite.angle)
        self.pendulum_sprite.center_y = self.pivot_sprite.center_y + PENDULUM_LENGTH * math.cos(
            self.pendulum_sprite.angle)
        # self.pendulum_sprite.center_y = self.pivot_sprite.center_y + self.line_sprite.width / 2 * math.cos(
        #     self.line_sprite.angle)

        # Check if the pendulum has reached a terminal state
        if self.pendulum_sprite.angle == 0 and self.pendulum_sprite.change_angle == 0:
            self.physics_engine.is_active = False
            print("Terminal state reached")

        # # Update the physics engine
        # self.physics_engine.step()
        #
        # # Update the line sprite
        # self.line_sprite.angle = self.pendulum_sprite.angle
        # self.line_sprite.center_x = self.pivot_sprite.center_x
        # self.line_sprite.center_y = self.pivot_sprite.center_y
        #
        # # Update the pendulum sprite
        # self.pendulum_sprite.center_x = self.pivot_sprite.center_x + self.line_sprite.width/2 * math.sin(self.line_sprite.angle)
        # self.pendulum_sprite.center_y = self.pivot_sprite.center_y + self.line_sprite.width/2 * math.cos(self.line_sprite.angle)
        #
        # # Check if the pendulum has reached a terminal state
        # if self.pendulum_sprite.angle == 0 and self.pendulum_sprite.angular_velocity == 0:
        #     self.physics_engine.is_active = False
        #     print("Terminal state reached")

        # print(self.pendulum_sprite.angle)
        # print(self.pendulum_sprite.angular_velocity)

    # def on_key_press(self, key, key_modifiers):
    #     """
    #     Called whenever a key on the keyboard is pressed.
    #
    #     For a full list of keys, see:
    #     https://api.arcade.academy/en/latest/arcade.key.html
    #     """
    #     pass
    #
    # def on_key_release(self, key, key_modifiers):
    #     """
    #     Called whenever the user lets off a previously pressed key.
    #     """
    #     pass
    #
    # def on_mouse_motion(self, x, y, delta_x, delta_y):
    #     """
    #     Called whenever the mouse moves.
    #     """
    #     pass
    #
    # def on_mouse_press(self, x, y, button, key_modifiers):
    #     """
    #     Called when the user presses a mouse button.
    #     """
    #     pass
    #
    # def on_mouse_release(self, x, y, button, key_modifiers):
    #     """
    #     Called when a user releases a mouse button.
    #     """
    #     pass


def main():
    """ Main function """
    game = MyPendulum(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    game.setup()
    arcade.run()


if __name__ == "__main__":
    main()

# import arcade
# import pymunk
# import math
#
# SCREEN_WIDTH = 800
# SCREEN_HEIGHT = 600
#
# class Pendulum(arcade.Window):
#     def __init__(self, width, height):
#         super().__init__(width, height, "Swinging Pendulum")
#         self.space = pymunk.Space()
#         self.space.gravity = (0, 10)  # Set the gravity
#         self.pendulum_length = 100
#         self.pendulum_mass = 10
#         self.angle = 0
#         self.pendulum_body = None
#         self.pendulum_shape = None
#
#     def setup(self):
#         # arcade.set_background_color(arcade.color.WHITE)
#         arcade.set_background_color(arcade.color.AMAZON)
#
#         # Create the pendulum body
#         moment_of_inertia = pymunk.moment_for_circle(self.pendulum_mass, 0, self.pendulum_length)
#         self.pendulum_body = pymunk.Body(self.pendulum_mass, moment_of_inertia)
#         self.pendulum_body.position = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + self.pendulum_length)
#         self.space.add(self.pendulum_body)
#
#         # Create the pendulum shape
#         self.pendulum_shape = pymunk.Circle(self.pendulum_body, self.pendulum_length)
#         self.pendulum_shape.friction = 0.5
#         self.space.add(self.pendulum_shape)
#
#     def on_draw(self):
#         arcade.start_render()
#
#         # Draw the pendulum bob
#         x, y = self.pendulum_body.position
#         arcade.draw_line(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, x, y, arcade.color.BLACK, 5)
#         arcade.draw_circle_filled(x, y, 10, arcade.color.RED)
#
#     def update(self, delta_time):
#         # Update the physics
#         self.space.step(delta_time)
#
#         # Update the pendulum angle
#         x, y = self.pendulum_body.position
#         self.angle = math.atan2(y - (SCREEN_HEIGHT // 2), x - (SCREEN_WIDTH // 2))
#
#     def on_key_press(self, key, modifiers):
#         if key == arcade.key.ESCAPE:
#             arcade.close_window()
#
# def main():
#     game = Pendulum(SCREEN_WIDTH, SCREEN_HEIGHT)
#     game.setup()
#     arcade.run()
#
# if __name__ == "__main__":
#     main()
