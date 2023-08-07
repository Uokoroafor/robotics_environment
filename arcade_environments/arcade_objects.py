import arcade
import pymunk
import math
from typing import Tuple, List, Optional


class Ball:
    def __init__(self, x: int, y: int, radius: float, collision_type: int = 1, elasticity: float = 0.8,
                 mass: float = 10.0):
        """ This is a ball object that can be used in any arcade environment. It is a wrapper around pymunk's circle object.

        Args:
            x (int): The x coordinate of the ball's center
            y (int): The y coordinate of the ball's center
            radius (float): The radius of the ball
            collision_type (int, optional): The collision type of the ball. Defaults to 1.
            elasticity (float, optional): The elasticity of the ball. Defaults to 0.8.
            mass (float, optional): The mass of the ball. Defaults to 10.0.
        """
        moment = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        self.body = pymunk.Body(mass, moment)
        self.body.position = pymunk.Vec2d(x, y)
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.elasticity = elasticity  # This means that the coin will bounce back with elasticity%
        # of its original speed
        self.radius = radius
        self.shape.collision_type = collision_type

    def draw(self):
        arcade.draw_circle_filled(self.body.position.x, self.body.position.y, self.radius, arcade.color.GOLD)


class Rectangle:
    def __init__(self, x: float, y: float, width: float, height: float, angle: float = 0, collision_type: int = 1):
        """ This is a rectangle object that can be used in any arcade environment. It is a wrapper around pymunk's segment object.

        Args:
            x (int): The x coordinate of the rectangle's center
            y (int): The y coordinate of the rectangle's center
            width (float): The width of the rectangle
            height (float): The height of the rectangle
            angle (float, optional): The angle of the rectangle. Defaults to 0.
            collision_type (int, optional): The collision type of the rectangle. Defaults to 1.
        """

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
        self.shape.collision_type = collision_type

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


class Triangle:

    def __init__(self, x: float, y: float, width: float, height: float, orientation: float = 0.0,
                 collision_type: int = 1, angles: Optional[List[float]] = None, elasticity: float = 1.0):
        """ This is a triangle object that can be used in any arcade environment. It is a wrapper around pymunk's segment object.

        Args:
            x (int): The x coordinate of the triangle's center
            y (int): The y coordinate of the triangle's center
            width (float): The width of the triangle
            height (float): The height of the triangle
            orientation (float, optional): The angle of the triangle. Defaults to 0.
            collision_type (int, optional): The collision type of the triangle. Defaults to 1.
            angles (List[float], optional): The angles of the triangle. Defaults to [0, 120, 240].
            elasticity (float, optional): The elasticity of the triangle. Defaults to 1.0.

        """

        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = pymunk.Vec2d(x, y)
        self.shape = pymunk.Segment(self.body, pymunk.Vec2d(-width / 2, -height / 2),
                                    pymunk.Vec2d(width / 2, height / 2), 1)
        self.shape.elasticity = elasticity
        self.body.orientation = math.radians(orientation)
        self.width = width
        self.height = height
        self.orientation = orientation
        self.center_x = self.body.position.x
        self.center_y = self.body.position.y
        self.color = arcade.color.BROWN
        self.top = self.center_y + self.height / 2
        self.bottom = self.center_y - self.height / 2
        self.shape.collision_type = collision_type
        if angles is None:
            self.angles = [0, 120, 240]

    def draw(self):
        # Use the width, height and angles to calculate the corners of the triangle
        corners = []
        for angle in self.angles:
            corners.append((self.center_x + self.width / 2 * math.cos(math.radians(angle)),
                            self.center_y + self.height / 2 * math.sin(math.radians(angle))))

        arcade.draw_polygon_filled(corners, self.color)


class Ramp(Triangle):

    def __init__(self, x: float, y: float, width: float, height: float, orientation: float = 0.0,
                 collision_type: int = 1, elasticity: float = 1.0):
        """ This is a ramp object that can be used in any arcade environment. It is a wrapper around pymunk's
        segment object. It is a right-angled triangle with angles [0, 90, 180].

        Args:
            x (int): The x coordinate of the ramp's center
            y (int): The y coordinate of the ramp's center
            width (float): The width of the ramp
            height (float): The height of the ramp
            orientation (float, optional): The angle of the ramp. Defaults to 0.
            collision_type (int, optional): The collision type of the ramp. Defaults to 1.
            angles (List[float], optional): The angles of the ramp. Defaults to [0, 120, 240].
            elasticity (float, optional): The elasticity of the ramp. Defaults to 1.0.
        """
        # want a right angled triangle
        angles = [0, 90, 180]
        super().__init__(x, y, width, height, orientation, collision_type, angles, elasticity)
        self.color = arcade.color.GREEN


