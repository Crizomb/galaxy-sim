from vec3 import Vec3
from typing import Union, List
from matplotlib import pyplot as plt
import time
import numpy as np
from vispy import scene, app


"""
Galaxy simulation using the Barnes-Hut algorithm.
following this guide http://arborjs.org/docs/barnes-hut#:~:text=The%20Barnes%2DHut%20algorithm%20is,of%20which%20may%20be%20empty)
"""

"""Global const"""
THETA = 0.5 #more is less precise but faster
THETA_SQUARED = THETA ** 2
CHILDREN_CONV = {"top_left_front": 0, "top_left_back": 1, "top_right_front": 2, "top_right_back": 3, "bottom_left_front": 4, "bottom_left_back": 5, "bottom_right_front": 6, "bottom_right_back": 7}
#is_top, is_left, is_front = pos.y > center.y, pos.x < center.x, pos.z < center.z
CHILDREN_CONV_BOOL = {(True, True, True): 0, (True, True, False): 1, (True, False, True): 2, (True, False, False): 3, (False, True, True): 4, (False, True, False): 5, (False, False, True): 6, (False, False, False): 7}
SMOOTHING = 0.5
G = 10

"""useful functions"""

def newton_law(mass1: float, mass2: float, vec_diff: Vec3) -> Vec3:
    """Return the force applied by mass1 on mass2."""
    return G * vec_diff * (mass1 * mass2 / (vec_diff.norm + SMOOTHING) ** 3)

def get_time(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.perf_counter() - start} seconds")
        return result

    return wrapper


"""Type defintion"""

class Star:
    __slots__ = "pos", "speed", "mass"
    def __init__(self, pos: Vec3, speed: Vec3, mass: float):
        self.pos = pos
        self.speed = speed
        self.mass = mass


class Node:
    """"Node class"""
     __slots__ = "center", "center_of_mass", "mass", "children", "width"
    def __init__(self, center: Vec3, center_of_mass: Vec3, mass: float, children, width: float):
        self.center = center
        self.center_of_mass = center_of_mass
        self.mass = mass
        self.children = children #[] if is_external else child 0 is top_left_front, 1 top_left_back, etc... see CHILDREN_CONV
        self.width = width


    @property
    def is_external(self) -> bool:
        """Return True if the node is external."""
        return self.children == []

    def is_empty(self) -> bool:
        """Return True if the node is empty."""
        return self.mass == 0

    def __repr__(self):
        if self.is_external:
            return f"External(center={self.center}, center_of_mass={self.center_of_mass}, mass={self.mass}, width={self.width}"
        else:
            return f"Internal(center={self.center}, center_of_mass={self.center_of_mass}, mass={self.mass}, width={self.width}"

    def get_child_index(self, pos: Vec3) -> int:
        """Return the index of the child in which the star should be inserted. see CHILDREN_CONV."""
        center = self.center
        is_top, is_left, is_front = pos.y > center.y, pos.x < center.x, pos.z < center.z
        return CHILDREN_CONV_BOOL[(is_top, is_left, is_front)]

    def create_children(self) -> None:
        """Create the children of a node. Divide the width by 2. Divide self into 8 children."""
        children = [None] * 8
        new_width = self.width / 2
        center = self.center
        for is_top in (True, False):
            for is_left in (True, False):
                for is_front in (True, False):
                    center_x = center.x - new_width if is_left else center.x + new_width
                    center_y = center.y + new_width if is_top else center.y - new_width
                    center_z = center.z - new_width if is_front else center.z + new_width
                    index = CHILDREN_CONV_BOOL[(is_top, is_left, is_front)]
                    children[index] = Node(Vec3(center_x, center_y, center_z), Vec3(0, 0, 0), 0, [], new_width)
        self.children = children

    def insert(self, star: Star) -> None:
        """Insert a star in the tree."""
        if self.is_empty():
            self.center_of_mass = star.pos
            self.mass = star.mass

        elif not self.is_external:
            self.center_of_mass = (self.center_of_mass * self.mass + star.pos * star.mass) / (self.mass + star.mass)
            self.mass += star.mass
            self.children[self.get_child_index(star.pos)].insert(star)

        else:
            self.create_children()
            self.center_of_mass = (self.center_of_mass * self.mass + star.pos * star.mass) / (self.mass + star.mass)
            self.children[self.get_child_index(star.pos)].insert(star)


    def get_force(self, star: Star) -> Vec3:
        """Return the force applied by the node on the star. Peformance bottleneck (called for each star)."""
        dist_squared = (self.center_of_mass - star.pos).norm_squared
        if self.is_external or self.width*self.width / dist_squared < THETA_SQUARED:
            return newton_law(self.mass, star.mass, self.center - star.pos)
        else:
            return sum((child.get_force(star) for child in self.children), Vec3.zero())


"""Init simulation"""
WIDTH = 50

"""Create random stars"""
def create_stars(nb_stars: int) -> List[Star]:
    """Return a galaxy like array of n_stars stars so stars rotate around the center, stars are more or less in the y=0 plan"""
    stars = []
    rand = lambda: np.random.normal() * WIDTH / 5
    for _ in range(nb_stars):
        x,y,z = rand(), rand() / 5, rand()
        pos = Vec3(x, y, z)
        orthog = Vec3(1, 0, -x / z) if z > 0 else Vec3(-1, 0, x / z)
        speed = orthog.normalize()*pos.norm
        stars.append(Star(pos, speed, 1))
        
    #add center blackhole
    stars.append(Star(Vec3(0, 0, 0), Vec3(0, 0, 0), nb_stars/10))
    
    return stars

def create_tree(stars: List[Star]) -> Node:
    """Return the root of the tree."""
    root = Node(Vec3(WIDTH / 2, WIDTH / 2, WIDTH / 2), Vec3(0, 0, 0), 0, [], WIDTH)
    for star in stars:
        root.insert(star)
    return root


def update_star(star, root, dt):
    """Update the speed and the position of a star.
    using leapfrog method."""
    force = root.get_force(star)
    star.speed += force * dt / 2
    star.pos += star.speed * dt
    force = root.get_force(star)
    star.speed += force * dt / 2


def update_speed_and_pos(stars: List[Star], root: Node, dt: float) -> None:
    """Update the speed and the position of the stars"""
    for star in stars:
        update_star(star, root, dt)



def compute_next_state(stars: List[Star], dt: float):
    """Compute the next state of the simulation."""
    root = create_tree(stars)
    update_speed_and_pos(stars, root, dt)

"""
Graphical part
"""
plt.style.use("dark_background")
def animate_stars_plt(stars_array, dt):
    """Animate the stars in the array. turn off the axis and the grid. fix the size of the figure."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    ax.grid(False)

    size = 20
    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)
    ax.set_zlim(-size, size)
    ax.set_aspect('equal')
    #set view parrallel to the xz plane
    ax.view_init(elev=90, azim=90)

    #plot the stars
    while True:
        start = time.perf_counter()
        ax.clear()
        ax.set_axis_off()
        ax.grid(False)
        ax.set_xlim(-size, size)
        ax.set_ylim(-size, size)
        ax.set_zlim(-size, size)
        ax.set_aspect('equal')
        compute_next_state(stars_array, dt)
        X, Y, Z = zip(*[star.pos for star in stars_array])
        ax.scatter(X, Y, Z, s=0.1, color='r')
        end = time.perf_counter()
        print(f"fps: {1/(end-start):}")
        plt.pause(0.01)


    plt.show()


def animate_stars_vispy(stars_array, dt):
    MAX_FPS = 20
    """Animate the stars in the array. faster than the matplotlib version."""
    canvas = scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    #center the view
    view.camera = 'turntable'
    view.camera.distance = 50

    @get_time
    def update(event):
        a = time.perf_counter()
        compute_next_state(stars_array, dt)
        points = np.array([star.pos.np for star in stars_array])
        scatter.set_data(points, edge_color=None, face_color=(1, 0.5, 1, 0.3), size=4, )
        canvas.update()
        b = time.perf_counter()
        if b-a < 1/MAX_FPS:
            time.sleep(1/MAX_FPS - (b-a))


    timer = app.Timer('auto', connect=update, start=True)
    scatter = scene.visuals.Markers()
    view.add(scatter)
    canvas.show()
    app.run()



def main():
    stars = create_stars(10**3)
    animate_stars_vispy(stars, 0.01)

if __name__ == '__main__':
    main()

























