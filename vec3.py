import numpy as np


Num = Union[int, float]

class Vec3:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float):
        self.x, self.y, self.z = x, y, z

    def __add__(self, other :"Vec3"):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __iadd__(self, other : "Vec3"):
        self.x, self.y, self.z = self.x + other.x, self.y + other.y, self.z + other.z
        return self

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    def __sub__(self, other : "Vec3"):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __isub__(self, other : "Vec3"):
        self.x, self.y, self.z = self.x - other.x, self.y - other.y, self.z - other.z
        return self

    def __mul__(self, other : Num):
        return Vec3(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other : Num):
        return Vec3(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other : Num):
        return Vec3(self.x / other, self.y / other, self.z / other)

    def __eq__(self, other : "Vec3"):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __ne__(self, other : "Vec3"):
        return not self == other

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __copy__(self):
        return Vec3(self.x, self.y, self.z)

    def __repr__(self):
        return f"Vec3({self.x}, {self.y}, {self.z})"

    @property
    def np(self):
        return np.array((self.x, self.y, self.z))

    @property
    def norm_squared(self) -> float:
        return self.x**2 + self.y**2 + self.z**2

    @property
    def norm(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2)**0.5

    def cross(self, other : "Vec3"):
        return Vec3(self.y * other.z - self.z * other.y, self.z * other.x - self.x * other.z, self.x * other.y - self.y * other.x)

    def dot(self, other : "Vec3"):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def normalize(self):
        return self / self.norm

    def rotate(self, axis : "Vec3", angle : float):
        axis = axis.normalize()
        a = axis * self.dot(axis)
        b = self - a
        c = axis.cross(b)
        return a + b * math.cos(angle) + c * math.sin(angle)

    @staticmethod
    def random():
        """Return a random vector. x, y and z are between 0 and 1."""
        return Vec3(random.random(), random.random(), random.random())

    @staticmethod
    def zero():
        return Vec3(0, 0, 0)




