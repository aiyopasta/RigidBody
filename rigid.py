# Aditya Abhyankar 2023
# TODO (NEXT): 1) Add semi-implicit Euler, 2) Show comparison of KE+PE conservation, 3) Text bells and whistles

from tkinter import *
import numpy as np
import time
np.set_printoptions(suppress=True)

# Window size. Note: 1920/2 = 960 will be the width of each half of the display (2D | 3D)
window_w = 1700
window_h = 1000

# Tkinter Setup
root = Tk()
root.title("Rigid Body Fun")
root.attributes("-topmost", True)
root.geometry(str(window_w) + "x" + str(window_h))  # window size hardcoded
w = Canvas(root, width=window_w, height=window_h)
w.pack()


# Coordinate Shift
def A(pt: np.ndarray):
    assert len(pt) == 2
    return (pt * np.array([1, -1])) + np.array([window_w/2, window_h/2])


def Ainv(pt: np.ndarray):
    assert len(pt) == 2
    return (pt - np.array([window_w/2, window_h/2])) * np.array([1, -1])


# Helper functions
# Rotation matrix from angle (must be in radians, of course)
def R(theta: float):
    return np.column_stack([[np.cos(theta), np.sin(theta)],
                            [-np.sin(theta), np.cos(theta)]])


class RigidBody:
    def __init__(self, points, init_xy=np.array([0.,0.]), init_v=np.array([0.,0.]), init_theta=0., init_omega=0., mass=1., inertia=500., id='none'):
        '''
            points: A python list of n numpy arrays of size 2 comprising the vertices of the body in cc order.
                    (IN BODY REFERENCE FRAME).
            init_x: A np array of size 2 denoting initial position.
            init_v: A np array of size 2 denoting initial velocity.
            init_theta: A double denoting initial rotation angle.
            init_omega: A double denoting initial angular velocity.

            Other params: mass, inertia (scalar as we're in 2D)
            ID: string for debugging
        '''

        self.points = [np.array([p[0], p[1]]) for p in points]
        self.centroid = self.compute_centroid()
        self.xy, self.v, self.theta, self.omega = init_xy, init_v, init_theta, init_omega
        self.m = mass
        self.I = inertia
        self.ID = id

    # Assume convex poly, split into triangles and take weighted avg of COM.
    # Fun fact: It's not at the mean of the positions of the vertices! Think about why this is by pondering the 1D case.
    # Note: COM is also in shape's reference frame!
    def compute_centroid(self):
        centroid = np.array([0., 0.])
        total_area = 0
        for i in range(len(self.points)-2):
            p0, p1, p2 = self.points[0], self.points[i+1], self.points[i+2]
            area = np.cross((p2 - p0), (p1 - p0)) / 2
            total_area += area
            centroid += ((p0 + p1 + p2) / 3.) * area

        return centroid / total_area

    def get_poly_points(self, screen_space=True):
        # Assume x, theta have already been updated
        points = []
        for p in self.points:
            # Rotate about center of mass
            loc = np.dot(R(self.theta), p - self.centroid) + self.centroid + self.xy
            if screen_space:
                points.extend(A(loc))
            else:
                points.extend(loc)

        return points

    def get_centroid_points(self, radius=5):
        return [*A(self.centroid - radius + self.xy),
                *A(self.centroid + radius + self.xy)]

    # Exact solutions for 2nd order ODE (linear or quadratic)
    def solve_exact(self, h, const_a=np.array([0,0]), const_alpha=0.):
        # The exact solution (from Taylor expansion of quadratic x(t)=x0+tv+0.5at^2 at t=tn+h,
        # where v is the initial velocity (constant) and a is the constant acceleration).
        self.xy += (h * self.v) + (np.power(h, 2) * const_a / 2.0)
        self.theta += (h * self.omega) + (np.power(h, 2) * const_alpha / 2.0)
        # Not necessary, but for energy computation / momentum calculation purposes
        self.v += h * const_a
        self.omega += h * const_alpha


# Get regular n-gon vertices
def regular_ngon_verts(n_sides, sidelen=50):
    points = []
    for theta in np.arange(0, 2*np.pi, 2*np.pi / n_sides):
        p = (sidelen * np.cos(theta), sidelen * np.sin(theta))
        points.append(p)

    return points


# Switches / Knobs 🎚
mode = 0   # 0 = Unconstrained rigid bodies (no forces)

# Bells and whistles 🔔
mode_text = ['Unconstrained (Integrator is exact)']
instructions = ['Click anywhere to spawn rigid body. \'G\' to toggle constant gravity.']

# Universal parameters 🌎
bodies = []
gravity = False

# Simulation params
dt = 1. / 60.  # 60 fps


# Main function
def run():
    global dt, mode, bodies, gravity
    w.configure(background='black')

    # Prep
    if mode == 0:
        pass


    while True:
        w.delete('all')
        # Update Model
        if mode == 0:
            for body in bodies:
                g = np.array([0.0, -100.0]) if gravity else np.array([0, 0])
                body.solve_exact(dt, const_a=g)
                if body.xy[1] < -window_h/2:
                    bodies.remove(body)


        # Paint scene
        # Show labels
        w.create_text(window_w/2, 40, text='Mode: '+str(mode_text[mode]), fill='white', font='Avenir 25')
        w.create_text(window_w/2, window_h-50, text=instructions[mode], fill='yellow', font='Avenir 20')
        # Mode specific painting
        if mode == 0:
            for body in bodies:
                w.create_polygon(*body.get_poly_points(), fill='red', outline='white')
                w.create_oval(body.get_centroid_points(), fill='blue')

        # End run
        w.update()
        time.sleep(dt)


# Key bind
def key_pressed(event):
    global gravity
    if mode == 0:
        if event.char == 'g':
            gravity = not gravity


# MOUSE + KEY METHODS ————————————————————————————————————————————————
def mouse_click(event):
    global mode, bodies
    clicked_pt = Ainv(np.array([event.x, event.y]))
    if mode == 0:
        rng_v = (np.random.random(2) - 0.5) * 300.0      # arbitrary max mag
        rng_theta = np.random.random() * 2 * np.pi      # arbitrary max mag
        rng_omega = (np.random.random() - 0.5) * 5.0    # arbitrary max mag
        rng_sides = np.random.randint(3, 11)            # arbitrary range
        verts = regular_ngon_verts(rng_sides) #+ np.ran
        bodies.append(RigidBody(verts, init_xy=clicked_pt, init_v=rng_v, init_theta=rng_theta, init_omega=rng_omega))


def mouse_release(event):
    # print('mouse released')
    pass


def left_drag(event):
    # print('mouse dragged')
    pass


def motion(event):
    # print('motion')
    pass


# Mouse bind
w.bind('<Motion>', motion)
w.bind("<Button-1>", mouse_click)
w.bind("<ButtonRelease-1>", mouse_release)
w.bind('<B1-Motion>', left_drag)


root.bind("<KeyPress>", key_pressed)
w.pack()


# Main function
if __name__ == '__main__':
    run()

# Necessary line for Tkinter
mainloop()