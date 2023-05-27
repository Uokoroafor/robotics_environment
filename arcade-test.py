import arcade
import random

WINDOW_SIZE = 500
SCREEN_BACKGROUND_COLOUR = (0, 10, 30)
STICK_RADIUS = 10
STICK_MASS = 10
STICK_FRICTION = 0.5
STICK_COLOUR = (200, 200, 200)
OBJECT_WIDTH = 20
OBJECT_HEIGHT = 40
OBJECT_MASS = 100
OBJECT_FRICTION = 0.5
OBJECT_DAMPING = 0.01
OBJECT_COLOUR = (120, 120, 150)
WALL_THICKNESS = 20
WALL_COLOUR = (10, 20, 50)
EPISODE_LENGTH = 100


class MainWindow(arcade.Window):

    def __init__(self, width, height):
        # Initialise
        super().__init__(width, height, "Arcade-Test")
        self.background_color = SCREEN_BACKGROUND_COLOUR
        self.global_time_step = 0
        self.episode_time_step = 0
        self.episode_velocity = (0, 0)

        # Create the stick
        self.stick_sprite = arcade.SpriteCircle(radius=STICK_RADIUS, color=STICK_COLOUR)
        self.stick_sprite.set_position(180, 180)

        # Create the object
        self.object_sprite = arcade.SpriteSolidColor(width=OBJECT_WIDTH, height=OBJECT_HEIGHT, color=OBJECT_COLOUR)
        self.object_sprite.set_position(200, 200)

        # Create the walls
        self.top_wall_sprite = arcade.SpriteSolidColor(width=WINDOW_SIZE, height=WALL_THICKNESS, color=WALL_COLOUR)
        self.top_wall_sprite.set_position(center_x=0.5 * WINDOW_SIZE, center_y=WINDOW_SIZE - 0.5 * WALL_THICKNESS)
        self.bottom_wall_sprite = arcade.SpriteSolidColor(width=WINDOW_SIZE, height=WALL_THICKNESS, color=WALL_COLOUR)
        self.bottom_wall_sprite.set_position(center_x=0.5 * WINDOW_SIZE, center_y=0.5 * WALL_THICKNESS)
        self.left_wall_sprite = arcade.SpriteSolidColor(width=WALL_THICKNESS, height=WINDOW_SIZE, color=WALL_COLOUR)
        self.left_wall_sprite.set_position(center_x=0.5 * WALL_THICKNESS, center_y=0.5 * WINDOW_SIZE)
        self.right_wall_sprite = arcade.SpriteSolidColor(width=WALL_THICKNESS, height=WINDOW_SIZE, color=WALL_COLOUR)
        self.right_wall_sprite.set_position(center_x=WINDOW_SIZE - 0.5 * WALL_THICKNESS, center_y=0.5 * WINDOW_SIZE)

        # Create the physics engine and add the stick and object and walls
        self.physics_engine = arcade.PymunkPhysicsEngine(gravity=(10, 0))
        self.physics_engine.add_sprite(sprite=self.stick_sprite, mass=STICK_MASS, friction=STICK_FRICTION,
                                       moment_of_inertia=9999999999, body_type=arcade.PymunkPhysicsEngine.DYNAMIC)
        self.physics_engine.add_sprite(sprite=self.object_sprite, mass=OBJECT_MASS, friction=OBJECT_FRICTION,
                                       damping=OBJECT_DAMPING, body_type=arcade.PymunkPhysicsEngine.DYNAMIC)
        self.physics_engine.add_sprite(sprite=self.top_wall_sprite, body_type=arcade.PymunkPhysicsEngine.STATIC)
        self.physics_engine.add_sprite(sprite=self.bottom_wall_sprite, body_type=arcade.PymunkPhysicsEngine.STATIC)
        self.physics_engine.add_sprite(sprite=self.right_wall_sprite, body_type=arcade.PymunkPhysicsEngine.STATIC)
        self.physics_engine.add_sprite(sprite=self.left_wall_sprite, body_type=arcade.PymunkPhysicsEngine.STATIC)

    def on_draw(self):
        self.clear()
        self.stick_sprite.draw()
        self.object_sprite.draw()
        self.top_wall_sprite.draw()
        self.bottom_wall_sprite.draw()
        self.left_wall_sprite.draw()
        self.right_wall_sprite.draw()

    def on_update(self, delta_time):
        self.physics_engine.set_velocity(self.stick_sprite, self.episode_velocity)
        self.physics_engine.step()
        self.global_time_step += 1
        self.episode_time_step += 1
        if self.episode_time_step == EPISODE_LENGTH:
            self.physics_engine.set_position(self.stick_sprite, (
            random.randint(50, WINDOW_SIZE - 50), random.randint(50, WINDOW_SIZE - 50)))
            self.episode_velocity = (random.randint(-50, 50), random.randint(-50, 50))
            self.episode_time_step = 0


def main():
    _ = MainWindow(WINDOW_SIZE, WINDOW_SIZE)
    arcade.run()


if __name__ == "__main__":
    main()
