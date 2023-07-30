# This environment will drop a ball from the center onto an angled shelf that the ball will bounce off.
# It will then log the x position of the ball at y = 0.

# Path: physics_environments/collision_dynamics.py
import arcade
import random
import math
import pymunk

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Collision Dynamics Test"

COIN_RADIUS = 10
COIN_SPEED_V = 0.5
COIN_SPEED_H = 10
COIN_RATE = 0.1
COIN_ELASTICITY = 0.5

SHELF_WIDTH = 100
SHELF_HEIGHT = 20

GRAVITY = (0, -100)
DAMPING = 0.9 # ADD WIND RESISTANCE

STARTING_SCORE = 10

# Collision types
COIN = 1
BOUNDARY = 2
SHELF = 3
GOAL = 4

BOUNDARY_WIDTH = 10
NUM_SHELVES = 1
SHELF_POSITION = [(SCREEN_WIDTH // 2, (SCREEN_HEIGHT // 2))]
class Coin:
    def __init__(self, x, y, radius):
        mass = 10
        moment = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        self.body = pymunk.Body(mass, moment)
        self.body.position = pymunk.Vec2d(x, y)
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.elasticity = COIN_ELASTICITY  # This means that the coin will bounce back with 80% of its original speed
        self.radius = radius
        self.shape.collision_type = COIN

    def draw(self):
        arcade.draw_circle_filled(self.body.position.x, self.body.position.y, self.radius, arcade.color.GOLD)


class Shelf:
    def __init__(self, x, y, width, height, angle=0):
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = pymunk.Vec2d(x, y)
        self.shape = pymunk.Segment(self.body, pymunk.Vec2d(-width / 2, -height / 2),
                                    pymunk.Vec2d(width / 2, height / 2), 1)
        self.shape.elasticity = 1.0
        self.body.angle = math.radians(angle)
        self.width = width
        self.height = height
        self.angle = angle
        self.center_x = self.body.position.x
        self.center_y = self.body.position.y
        self.color = arcade.color.BROWN
        self.top = self.center_y + self.height / 2
        self.bottom = self.center_y - self.height / 2
        self.shape.collision_type = SHELF

    def draw(self):
        half_width = self.width / 2
        half_height = self.height / 2

        # Define the four corners of the rectangle.
        corners = [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height)
        ]

        # Rotate each corner around the center of the rectangle.
        corners = [(x * math.cos(math.radians(self.angle)) - y * math.sin(math.radians(self.angle)),
                    x * math.sin(math.radians(self.angle)) + y * math.cos(math.radians(self.angle)))
                   for x, y in corners]

        # Offset each corner by the position of the rectangle.
        corners = [(x + self.center_x, y + self.center_y) for x, y in corners]

        arcade.draw_polygon_filled(corners, self.color)



class ShelfBounce(arcade.Window):
    def __init__(self, width, height, title, render=True):
        super().__init__(width, height, title)
        self.bounce_log = []
        arcade.set_background_color(arcade.color.WHITE)
        self.space = pymunk.Space()
        self.space.gravity = GRAVITY
        self.render = render


        self.coin_list = []
        self.shelf_list = []

        # Want goal state to be a line

        # self.space.add(self.goal_state.body, self.goal_state.shape)

        self.score = STARTING_SCORE

        # Create boundaries
        boundary_thickness = BOUNDARY_WIDTH
        boundaries = [
            # pymunk.Segment(self.space.static_body, (0, 0), (0, SCREEN_HEIGHT), boundary_thickness),  # Left
            # pymunk.Segment(self.space.static_body, (0, SCREEN_HEIGHT), (SCREEN_WIDTH, SCREEN_HEIGHT),
            #                boundary_thickness),  # Top
            # pymunk.Segment(self.space.static_body, (SCREEN_WIDTH, SCREEN_HEIGHT), (SCREEN_WIDTH, 0),
            #                boundary_thickness),  # Right
            # pymunk.Segment(self.space.static_body, (-SCREEN_WIDTH, 0), (2*SCREEN_WIDTH, 0), boundary_thickness)  # Bottom
        ]
        for boundary in boundaries:
            boundary.elasticity = 1.0
            boundary.friction = 1.0
            boundary.collision_type = BOUNDARY
            # print boundary positions
            # print(f"Boundary at ({boundary.a.x}, {boundary.a.y}) to ({boundary.b.x}, {boundary.b.y})")
        self.space.add(*boundaries)

        # Create collision handler
        handler = self.space.add_collision_handler(COIN, BOUNDARY)
        handler.begin = self.on_coin_boundary_collision

        handler = self.space.add_collision_handler(COIN, SHELF)
        handler.begin = self.on_coin_shelf_collision

        # handler = self.space.add_collision_handler(COIN, GOAL)
        # handler.begin = self.on_coin_goal_collision


    def generate_shelves(self):
        for _, (x, y) in enumerate(SHELF_POSITION):
            # We want no overlap between shelves
            angle = random.uniform(-45, 45)
            shelf = Shelf(x, y, SHELF_WIDTH, SHELF_HEIGHT, angle)
            self.space.add(shelf.body, shelf.shape)
            self.shelf_list.append(shelf)
            self.shelf_list.append(shelf)
            message = f"Shelf generated at ({shelf.center_x}, {shelf.center_y}) with angle {shelf.angle:.2f} degrees, width {shelf.width:.2f} and height {shelf.height}"
            # add message to log
            self.bounce_log.append(message)
            # print(message)



    def drop_coin(self):
        # Want the coin to be generated at the top of the screen in the middle 80% of the screen
        x = SCREEN_WIDTH // 2
        y = SCREEN_HEIGHT - COIN_RADIUS * 1.5 - BOUNDARY_WIDTH
        print(f"Coin generated at ({x}, {y})")
        self.bounce_log.append(f"Coin generated at ({x}, {y})")

        coin = Coin(x, y, COIN_RADIUS)
        self.space.add(coin.body, coin.shape)
        self.coin_list.append(coin)

    def on_draw(self):
        if self.render:
            arcade.start_render()
        # self.goal_state.draw()

            for coin in self.coin_list:
                coin.draw()

            for shelf in self.shelf_list:
                shelf.draw()

        # arcade.draw_text(f"Score: {self.score}", 10, 10, arcade.color.BLACK, 18)
        arcade.draw_text(f"Ball Drop on Shelf Angle: {self.shelf_list[0].angle:.2f} degrees", 10, 10, arcade.color.BLACK, 18)

    def update(self, delta_time):
        self.space.step(1 / 60.0)  # 60 fps
        for i, coin in enumerate(self.coin_list):
            # interpolate to find the x position at y=0 using gravity and velocity
            # s = ut + 1/2 at^2
            if coin.body.position.y < 0 < -(coin.body.position.y*delta_time + 0.5 * GRAVITY[1] * delta_time**2):
                # Use linear interpolation to estimate the x position at y=0
                fraction_of_time_step = -coin.body.position.y / (coin.body.velocity.y * delta_time)
                x_at_y_equals_zero = coin.body.position.x + coin.body.velocity.x * delta_time * fraction_of_time_step
                # print(f'Coin hits the ground at x = {x_at_y_equals_zero:.2f}.')
                self.bounce_log.append(f' Answer: Coin hits the ground at x = {x_at_y_equals_zero:.2f}.')
                arcade.close_window()

    def on_key_press(self, key, modifiers):
        #pass
        if key == arcade.key.SPACE:
            # Want the coin to be generated at the top of the screen in the middle 80% of the screen
            x = SCREEN_WIDTH // 2
            y = SCREEN_HEIGHT - COIN_RADIUS * 1.5 - BOUNDARY_WIDTH
            print(f"Coin generated at ({x}, {y})")

            coin = Coin(x, y, COIN_RADIUS)
            self.space.add(coin.body, coin.shape)
            self.coin_list.append(coin)

    def check_shelf_overlap(self, shelf1, shelf2):
        # Check if shelf1 overlaps with shelf2
        return (shelf1.center_x - shelf1.width / 2 < shelf2.center_x < shelf1.center_x + shelf1.width / 2 and
                shelf1.center_y - shelf1.height / 2 < shelf2.center_y < shelf1.center_y + shelf1.height / 2)

    def check_shelf_overlaps(self, shelf1):
        if len(self.shelf_list) > 1:
            for shelf2 in self.shelf_list:
                if shelf1 != shelf2:
                    if self.check_shelf_overlap(shelf1, shelf2):
                        return True
        return False

    def on_coin_boundary_collision(self, arbiter, space, data):
        return True

    def on_coin_shelf_collision(self, arbiter, space, data):
        # print("Coin hit shelf")
        self.bounce_log.append("Coin hit shelf")
        return True



def main():
    game = ShelfBounce(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    game.generate_shelves()
    # game.drop_coin()
    arcade.run()
    return game.bounce_log


if __name__ == "__main__":
    all_messages = []
    for i in range(10):
        message = main()
        message = ''.join(message)
        all_messages.append(message)
        # print(i)
    # remove duplicates
    # all_messages = list(set(all_messages))
    #
    # # Write all messages to a text file
    # with open("bounce_log.txt", "w") as f:
    #     for message in all_messages:
    #         f.write(message + "\n")