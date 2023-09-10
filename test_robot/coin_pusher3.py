import arcade
import random
import math
import pymunk

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Coin Pusher"

COIN_RADIUS = 10
COIN_SPEED_V = 0.5
COIN_SPEED_H = 0.5
COIN_RATE = 0.1

SHELF_WIDTH = 100
SHELF_HEIGHT = 20

GRAVITY = (0, -0.1)
BOUNCE_FACTOR = 0.8
PUSH_FORCE = 0.8

STARTING_SCORE = 10

# Collision types
COIN = 1
BOUNDARY = 2
SHELF = 3
PUSHER = 4

BOUNDARY_WIDTH = 10


class Coin:
    def __init__(self, x, y, radius):
        mass = 1
        moment = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        self.body = pymunk.Body(mass, moment)
        self.body.position = pymunk.Vec2d(x, y)
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.elasticity = 0.95
        self.change_x = random.uniform(-COIN_SPEED_H, COIN_SPEED_H)
        self.change_y = 0
        self.radius = radius
        self.center_x = self.body.position.x
        self.center_y = self.body.position.y
        self.bottom = self.center_y - self.radius
        self.top = self.center_y + self.radius
        self.shape.collision_type = COIN

    def update(self):
        self.change_y += GRAVITY[1]
        self.change_x += GRAVITY[0]
        self.body.position += pymunk.Vec2d(self.change_x, self.change_y)

        self.center_x = self.body.position.x
        self.center_y = self.body.position.y

        # Update top and bottom
        self.top = self.center_y + self.radius
        self.bottom = self.center_y - self.radius

    def draw(self):
        arcade.draw_circle_filled(
            self.center_x, self.center_y, self.radius, arcade.color.GOLD
        )


class Shelf:
    def __init__(self, x, y, width, height, angle=0):
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = pymunk.Vec2d(x, y)
        self.shape = pymunk.Segment(
            self.body,
            pymunk.Vec2d(-width / 2, -height / 2),
            pymunk.Vec2d(width / 2, height / 2),
            1,
        )
        self.shape.elasticity = 0.95
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
            (-half_width, half_height),
        ]

        # Rotate each corner around the center of the rectangle.
        corners = [
            (
                x * math.cos(math.radians(self.angle))
                - y * math.sin(math.radians(self.angle)),
                x * math.sin(math.radians(self.angle))
                + y * math.cos(math.radians(self.angle)),
            )
            for x, y in corners
        ]

        # Offset each corner by the position of the rectangle.
        corners = [(x + self.center_x, y + self.center_y) for x, y in corners]

        arcade.draw_polygon_filled(corners, self.color)


class Pusher:
    def __init__(self, x, y, width, height):
        self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.body.position = pymunk.Vec2d(x, y)
        self.shape = pymunk.Segment(
            self.body,
            pymunk.Vec2d(-width / 2, -height / 2),
            pymunk.Vec2d(width / 2, height / 2),
            1,
        )
        self.shape.elasticity = 0.95
        self.width = width
        self.height = height
        self.center_x = self.body.position.x
        self.center_y = self.body.position.y
        self.color = arcade.color.RED
        self.top = self.center_y + self.height / 2
        self.bottom = self.center_y - self.height / 2
        self.left = self.center_x - self.width / 2
        self.right = self.center_x + self.width / 2

        self.shape.collision_type = PUSHER

    def draw(self):
        # arcade.draw_rectangle_filled(self.center_x, self.center_y, self.width, self.height, self.color)
        arcade.draw_rectangle_filled(
            self.body.position.x,
            self.body.position.y,
            self.width,
            self.height,
            self.color,
        )


class CoinPusher(arcade.Window):
    def __init__(self, width, height, title):
        super().__init__(width, height, title)

        arcade.set_background_color(arcade.color.WHITE)
        self.space = pymunk.Space()
        self.space.gravity = GRAVITY

        self.coin_list = []
        self.shelf_list = []

        self.pusher = Pusher(width // 2, height // 6, 100, 50)
        self.space.add(self.pusher.body, self.pusher.shape)

        self.score = STARTING_SCORE

        # Create boundaries
        boundary_thickness = BOUNDARY_WIDTH
        boundaries = [
            pymunk.Segment(
                self.space.static_body, (0, 0), (0, SCREEN_HEIGHT), boundary_thickness
            ),  # Left
            pymunk.Segment(
                self.space.static_body,
                (0, SCREEN_HEIGHT),
                (SCREEN_WIDTH, SCREEN_HEIGHT),
                boundary_thickness,
            ),  # Top
            pymunk.Segment(
                self.space.static_body,
                (SCREEN_WIDTH, SCREEN_HEIGHT),
                (SCREEN_WIDTH, 0),
                boundary_thickness,
            ),  # Right
            pymunk.Segment(
                self.space.static_body, (SCREEN_WIDTH, 0), (0, 0), boundary_thickness
            ),  # Bottom
        ]
        for boundary in boundaries:
            boundary.elasticity = 1.0
            boundary.friction = 1.0
            boundary.collision_type = BOUNDARY
            # print boundary positions
            print(
                f"Boundary at ({boundary.a.x}, {boundary.a.y}) to ({boundary.b.x}, {boundary.b.y})"
            )
        self.space.add(*boundaries)

        # Create collision handler
        handler = self.space.add_collision_handler(COIN, BOUNDARY)
        handler.begin = self.on_coin_boundary_collision

        handler = self.space.add_collision_handler(COIN, SHELF)
        handler.begin = self.on_coin_shelf_collision

        handler = self.space.add_collision_handler(COIN, PUSHER)
        handler.begin = self.on_coin_pusher_collision

    def generate_shelves(self):
        for _ in range(2):
            # We want no overlap between shelves
            overlaps = True
            shelf = None
            while overlaps:
                x = random.uniform(0, SCREEN_WIDTH - SHELF_WIDTH)
                y = random.uniform(SCREEN_HEIGHT // 2, SCREEN_HEIGHT - SHELF_HEIGHT)
                angle = random.uniform(-45, 45)
                shelf = Shelf(x, y, SHELF_WIDTH, SHELF_HEIGHT, angle)
                self.space.add(shelf.body, shelf.shape)
                self.shelf_list.append(shelf)
                overlaps = self.check_shelf_overlaps(shelf)
            self.shelf_list.append(shelf)

    def on_draw(self):
        arcade.start_render()
        self.pusher.draw()

        for coin in self.coin_list:
            coin.draw()

        for shelf in self.shelf_list:
            shelf.draw()

        arcade.draw_text(f"Score: {self.score}", 10, 10, arcade.color.BLACK, 18)

    def update(self, delta_time):
        for coin in self.coin_list:
            coin.update()
            # coin.draw()
            # while True:  # Game loop
            self.space.step(1 / 60.0)  # Assume 60 fps

    def on_key_press(self, key, modifiers):
        if key == arcade.key.SPACE:
            # Want the coin to be generated at the top of the screen
            x = random.randint(0, SCREEN_WIDTH - self.pusher.width)
            y = SCREEN_HEIGHT - COIN_RADIUS * 1.5 - BOUNDARY_WIDTH
            print(f"Coin generated at ({x}, {y})")

            coin = Coin(x, y, COIN_RADIUS)
            self.space.add(coin.body, coin.shape)
            self.coin_list.append(coin)

    def on_mouse_motion(self, x, y, dx, dy):
        self.pusher.body.position = pymunk.Vec2d(x, self.pusher.center_y)
        self.pusher.draw()

    def check_shelf_overlap(self, shelf1, shelf2):
        # Check if shelf1 overlaps with shelf2
        return (
            shelf1.center_x - shelf1.width / 2
            < shelf2.center_x
            < shelf1.center_x + shelf1.width / 2
            and shelf1.center_y - shelf1.height / 2
            < shelf2.center_y
            < shelf1.center_y + shelf1.height / 2
        )

    def check_shelf_overlaps(self, shelf1):
        if len(self.shelf_list) > 1:
            for shelf2 in self.shelf_list:
                if shelf1 != shelf2:
                    if self.check_shelf_overlap(shelf1, shelf2):
                        return True
        return False

    @staticmethod
    def check_for_collision(coin_, shelf_):
        # Check if the bottom of the coin is touching the top of the shelf
        return (
            shelf_.center_x - shelf_.width / 2
            < coin_.center_x
            < shelf_.center_x + shelf_.width / 2
            and shelf_.center_y - shelf_.height / 2
            < coin_.center_y
            < shelf_.center_y + shelf_.height / 2
        )

    def check_for_collision_with_pusher(self, coin_):
        pusher_width = self.pusher.width
        pusher_height = self.pusher.height
        return (
            self.pusher.center_x - pusher_width / 2
            < coin_.center_x
            < self.pusher.center_x + pusher_width / 2
            and self.pusher.center_y - pusher_height / 2
            < coin_.center_y
            < self.pusher.center_y + pusher_height / 2
        )

    def on_coin_boundary_collision(self, arbiter, space, data):
        coin_shape, boundary_shape = arbiter.shapes

        # Remove the coin from the physics space
        self.space.remove(coin_shape.body, coin_shape)
        self.score -= 1
        if self.score < 0:
            # Game over
            arcade.close_window()
        print("Coin hit boundary")

        return True

    def on_coin_shelf_collision(self, arbiter, space, data):
        coin_shape, shelf_shape = arbiter.shapes
        coin = coin_shape.body
        shelf = shelf_shape.body

        # coin.change_y *= -1 * BOUNCE_FACTOR
        # coin.change_x -= math.sin(math.radians(shelf.angle)) * coin.change_y * BOUNCE_FACTOR
        coin.velocity = pymunk.Vec2d(
            coin.velocity.x
            - math.sin(math.radians(shelf.angle)) * coin.velocity.y * BOUNCE_FACTOR,
            coin.velocity.y * -1 * BOUNCE_FACTOR,
        )
        print("Coin hit shelf")

        return True

    def on_coin_pusher_collision(self, arbiter, space, data):
        coin_shape, pusher_shape = arbiter.shapes
        coin = coin_shape.body
        pusher = pusher_shape.body

        coin.velocity = pymunk.Vec2d(coin.velocity.x, coin.velocity.y * -1 * PUSH_FORCE)

        print("Coin hit pusher")
        return True


def main():
    game = CoinPusher(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    game.generate_shelves()
    arcade.run()


if __name__ == "__main__":
    main()
