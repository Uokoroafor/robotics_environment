import arcade
import random
import math
import pymunk

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Coin Pusher"

COIN_RADIUS = 10
COIN_SPEED_V = 0.5
COIN_SPEED_H = 10
COIN_RATE = 0.1

SHELF_WIDTH = 100
SHELF_HEIGHT = 20

GRAVITY = (0, -100)
DAMPING = 0.9 # ADD WIND RESISTANCE
BOUNCE_FACTOR = 0.8
PUSH_FORCE = 0.8

STARTING_SCORE = 10

# Collision types
COIN = 1
BOUNDARY = 2
SHELF = 3
GOAL = 4

BOUNDARY_WIDTH = 10
NUM_SHELVES = 2
SHELF_POSITIONS = [(SCREEN_WIDTH // 4, (3 * (SCREEN_HEIGHT // 4))), (3 * (SCREEN_WIDTH // 4), 3 * (SCREEN_HEIGHT // 4))]

class Coin:
    def __init__(self, x, y, radius):
        mass = 1
        moment = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        self.body = pymunk.Body(mass, moment)
        self.body.position = pymunk.Vec2d(x, y)
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.elasticity = 0.8  # This means that the coin will bounce back with 80% of its original speed
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


class GoalState:

    def __init__(self, x, y, width, height):
        self.body = pymunk.Body(
            body_type=pymunk.Body.STATIC)  # Kinematic bodies are not affected by gravity and they can be moved manually
        self.body.position = pymunk.Vec2d(x, y)
        self.shape = pymunk.Segment(self.body, pymunk.Vec2d(-width / 2, -height / 2),
                                    pymunk.Vec2d(width / 2, height / 2), 1)
        self.shape.elasticity = 0.95
        self.width = width
        self.height = height
        self.color = arcade.color.RED

        self.shape.collision_type = GOAL

    def draw(self):
        arcade.draw_rectangle_filled(self.body.position.x, self.body.position.y, self.width, self.height, self.color)


class CoinPusher(arcade.Window):
    def __init__(self, width, height, title):
        super().__init__(width, height, title)

        arcade.set_background_color(arcade.color.WHITE)
        self.space = pymunk.Space()
        self.space.gravity = GRAVITY


        self.coin_list = []
        self.shelf_list = []

        # Want goal state to be a line

        self.goal_state = GoalState(width // 2, height // 10, 100, 5)
        self.space.add(self.goal_state.body, self.goal_state.shape)

        self.score = STARTING_SCORE

        # Create boundaries
        boundary_thickness = BOUNDARY_WIDTH
        boundaries = [
            pymunk.Segment(self.space.static_body, (0, 0), (0, SCREEN_HEIGHT), boundary_thickness),  # Left
            pymunk.Segment(self.space.static_body, (0, SCREEN_HEIGHT), (SCREEN_WIDTH, SCREEN_HEIGHT),
                           boundary_thickness),  # Top
            pymunk.Segment(self.space.static_body, (SCREEN_WIDTH, SCREEN_HEIGHT), (SCREEN_WIDTH, 0),
                           boundary_thickness),  # Right
            pymunk.Segment(self.space.static_body, (SCREEN_WIDTH, 0), (0, 0), boundary_thickness)  # Bottom
        ]
        for boundary in boundaries:
            boundary.elasticity = 1.0
            boundary.friction = 1.0
            boundary.collision_type = BOUNDARY
            # print boundary positions
            print(f"Boundary at ({boundary.a.x}, {boundary.a.y}) to ({boundary.b.x}, {boundary.b.y})")
        self.space.add(*boundaries)

        # Create collision handler
        handler = self.space.add_collision_handler(COIN, BOUNDARY)
        handler.begin = self.on_coin_boundary_collision

        handler = self.space.add_collision_handler(COIN, SHELF)
        handler.begin = self.on_coin_shelf_collision

        handler = self.space.add_collision_handler(COIN, GOAL)
        handler.begin = self.on_coin_goal_collision

    def generate_shelves(self):
        for i, (x, y) in enumerate(SHELF_POSITIONS):
            # We want no overlap between shelves
            angle = random.uniform(-45, 45)
            shelf = Shelf(x, y, SHELF_WIDTH, SHELF_HEIGHT, angle)
            self.space.add(shelf.body, shelf.shape)
            self.shelf_list.append(shelf)
            self.shelf_list.append(shelf)
            print(f"Shelf {i + 1} at ({shelf.center_x}, {shelf.center_y}) with angle {shelf.angle:.2f} degrees, "
                  f"width {shelf.width:.2f} and height {shelf.height}")

    def on_draw(self):
        arcade.start_render()
        self.goal_state.draw()

        for coin in self.coin_list:
            coin.draw()

        for shelf in self.shelf_list:
            shelf.draw()

        # arcade.draw_text(f"Score: {self.score}", 10, 10, arcade.color.BLACK, 18)
        arcade.draw_text(f"{10-len(self.coin_list)} coin(s) left to drop", 10, 10, arcade.color.BLACK, 18)

    def update(self, delta_time):
        self.space.step(1 / 60.0)
        for i, coin in enumerate(self.coin_list):
            if random.randint(1, 100) < 5:
                print(f"Coin {i + 1} at ({coin.body.position.x:.2f}, {coin.body.position.y:.2f})")
                print(f"Coin {i + 1} velocity ({coin.body.velocity.x:.2f}, {coin.body.velocity.y:.2f})")

    def on_key_press(self, key, modifiers):
        if key == arcade.key.SPACE:
            # Want the coin to be generated at the top of the screen in the middle 80% of the screen
            x = random.randint(SCREEN_WIDTH // 10, SCREEN_WIDTH * 9 // 10)
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
        coin_shape, boundary_shape = arbiter.shapes

        # Remove the coin from the physics space
        self.space.remove(coin_shape.body, coin_shape)

        # Remove the coin from the coin list
        for coin in self.coin_list:
            if coin.shape == coin_shape:
                self.coin_list.remove(coin)
                break
        self.score -= 1
        if self.score < 0:
            # Game over
            arcade.close_window()
        print("Coin hit boundary")

        return True

    def on_coin_shelf_collision(self, arbiter, space, data):
        print("Coin hit shelf")

        return True

    def on_coin_goal_collision(self, arbiter, space, data):
        coin_shape, goal_shape = arbiter.shapes
        print('GOAL REACHED')
        arcade.close_window()

        return True


def main():
    game = CoinPusher(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    game.generate_shelves()
    arcade.run()


if __name__ == "__main__":
    main()
