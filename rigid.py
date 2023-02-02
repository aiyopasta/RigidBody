# Aditya Abhyankar 2023
# TODO (NEXT): 1) Add semi-implicit Euler (prediction: it'll just accelerate at twice the speed... but it'll be cool to see just how wrong I always was), 2) Show comparison of KE+PE conservation, 3) Text bells and whistles
#       AFTER: 1) Spring forces applied to arbitrary points on body (clickable), then 2) 2-body problem (Forward euler vs. Verlet vs. Backwards Euler vs. Semi-implicit euler)

from tkinter import *
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
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
    def __init__(self, points, init_xy=np.array([0.,0.]), init_v=np.array([0.,0.]), init_theta=0., init_omega=0., mass=1., inertia=500., solver_id=0, id='none'):
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
        self.xy_0, self.v_0, self.theta_0, self.omega_0 = copy.copy(self.xy), copy.copy(self.v), copy.copy(self.theta), copy.copy(self.omega)
        self.m = mass
        self.I = inertia
        self.ID = id

        self.n = 0  # number of times solved (useful for t0 ODE solver)
        self.solver_id = solver_id  # 0 = exact from tk, 1 = exact from t0, 2 = si-euler naive, 3 = si-euler expanded

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
            xy = Ainv(A(self.xy) % [window_w, window_h])
            loc = np.dot(R(self.theta), p - self.centroid) + self.centroid + xy
            if screen_space:
                points.extend(A(loc))
            else:
                points.extend(loc)

        return points

    def get_centroid_points(self, radius=5):
        xy = Ainv(A(self.xy) % [window_w, window_h])
        return [*A(self.centroid - radius + xy),
                *A(self.centroid + radius + xy)]

    def solve(self, h, const_a=np.array([0,0]), const_alpha=0.):
        params = [h, const_a, const_alpha]
        if self.solver_id == 0:
            self.solve_forward_euler(*params)
        elif self.solver_id == 1:
            self.solve_exact_tk(*params)
        elif self.solver_id == 2:
            self.solve_exact_t0(*params)
        elif self.solver_id == 3:
            self.solve_sieuler_naive(*params)
        else:
            self.solve_sieuler_expanded(*params)

    # BELOW LIE ALL THE EXPLICIT INTEGRATORS (THEY USE ACCELERATION INFO FROM THE PREVIOUS FRAME!
    # —————————————————————————————————————————————————————————————————————————————————————————————
    # 1) Meme solution for 2nd or 1st order ODE —— Euler's Method
    def solve_forward_euler(self, h, const_a=np.array([0,0]), const_alpha=0.):
        # Here, it's a meme method because we use the OLD vel instead of computing a new one,
        # which means (as you know from contrasting with the expanded semi-implicit Euler)
        # that we're only using a bad FIRST order guess here. Even though s.i. does not make a
        # proper first order guess either, it's at least second order as a poly in h. (It's also symplectic.)
        self.xy += h * self.v         # Use the old velocity
        self.theta += h * self.omega  # And the old omega
        self.v += h * const_a
        self.omega += h * const_alpha  # (And obviously old accelerations, as this is an explicit method)

    # 2) Exact solution for 2nd or 1st order ODE (linear or quadratic), using Taylor expansion at CURRENT TIME STEP.
    #    This is also the TRAPEZOID RULE (2nd order) Runge Kutta method! (The one we used in 562 as well.)
    def solve_exact_tk(self, h, const_a=np.array([0,0]), const_alpha=0.):
        # Exact solution from truncated Taylor expansion of x at t_{k+1} = t_k + h, namely:
        # x(t_{k+1}) = x(t_k) + h*xdot(t_k) + h^2/2! * xddot(t_k).
        self.xy += (h * self.v) + (np.power(h, 2) * const_a / 2.0)
        self.theta += (h * self.omega) + (np.power(h, 2) * const_alpha / 2.0)
        # In order to do the above, we need access to the previous step's derivative values,
        # i.e. xdot(t_k), which we now precompute exactly as well: v(t_{k+1}) = v(t_k) + h*vdot(t_k).
        # Note 1: This v(t_{k+1}) value /becomes/ the v(t_k) in the next iteration!
        # Note 2: If the acceleration is 0, this value just remains constant! There's actually no need for computation!
        self.v += h * const_a
        self.omega += h * const_alpha

    # 3) Exact solution for 2nd or 1st order ODE (linear or quadratic), using Taylor expansion at INITIAL TIME STEP.
    def solve_exact_t0(self, h, const_a=np.array([0,0]), const_alpha=0.):
        self.n += 1
        elapsed_t = h * self.n
        # Exact solution from truncated Taylor expansion of x at t_0, namely:
        # x(t_0 + delta_t) = x(t_0) + (delta_t)*xdot(t_0) + (delta_t)^2/2! * xddot(t_0).
        self.xy = self.xy_0 + (elapsed_t * self.v_0) + (np.power(elapsed_t, 2.0) * const_a / 2)
        self.theta = self.theta_0 + (elapsed_t * self.omega_0) + (np.power(elapsed_t, 2.0) * const_alpha / 2)
        # Now the /following/ two updates are not necessary for the above!
        # But we'll compute them anyway in case we need them for energies and stuff, namely
        # v(t_0 + delta_t) = v(t_0) + (delta_t * vdot(t_0))
        self.v = self.v_0 + (elapsed_t * const_a)
        self.omega = self.omega_0 + (elapsed_t * const_alpha)

    # 4) The naive, inexact update everyone (including me) usually does when acceleration is constant.
    #    S.I. = "semi-implicit Euler"
    def solve_sieuler_naive(self, h, const_a=np.array([0,0]), const_alpha=0.):
        self.v += h * const_a
        self.xy += h * self.v  # = (self.v_old * h) + (h^2 * const_a), which is inexact (missing a factor of 1/2!).
        self.omega += h * const_alpha
        self.theta += h * self.omega

    # 5) Expanded version of the naive, inexact update that everyone (including me) does when acceleration is constant.
    #    S.I. = "semi-implicit Euler"
    def solve_sieuler_expanded(self, h, const_a=np.array([0,0]), const_alpha=0.):
        self.xy += (h * self.v) + (np.power(h, 2.0) * const_a)  # inexact, missing factor of 1/2!
        self.theta += (h * self.omega) + (np.power(h, 2.0) * const_alpha)
        # Precompute new velocity for use in next iteration
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
mode_text = ['EXPERIMENT #1: Constant Velocity / Acceleration Integration', 'EXPERIMENT #2: Non-constant Force Integration']
mode_subtext = ['Constant Velocity: All will tie. Constant Acceleration: Semi-implicit gains extra energy and wins, while Euler strays behind the actual solution!', 'Comparison between: 1) Forward Euler (Meme), 2) Verlet (Explicit), 3) Backward Euler (Pixar), 3) Semi-Implicit (Explicit, really)']
instructions = ['Click anywhere to spawn 5 rigid bodies, one per integrator. \'G\' to toggle between space-mode and competition-mode.', 'Instructions: idk lol']

# Mode 0 parameters
gravity = True
frame = 0
max_frame = 400

# Mode 1 parameters
G = 6.67E-11  # gravitational constant

# Universal parameters 🌎
bodies = []

# Simulation params
dt = 1. / 60.  # 60 fps


# Main function
def run():
    global dt, frame, max_frame, mode, bodies, gravity, mode_subtext
    w.configure(background='black')

    # Prep
    if mode == 0:
        pass

    while True:
        w.delete('all')
        if len(bodies) > 0 and gravity:
            frame += 1

        # Update Model
        if mode == 0:
            for i, body in enumerate(bodies):
                g = np.array([0, 0])
                if gravity:
                    g = np.array([0.0, -1000.0])
                    if frame < max_frame:
                        body.solve(dt, const_a=g)
                    elif frame > max_frame + 100:
                        bodies.clear()
                        frame = 0
                        break
                else:
                    body.solve(dt, const_a=g)
                    offscreen = abs(body.xy[0]) >= window_w/2 or abs(body.xy[1]) >= window_h/2
                    if offscreen:
                        bodies.remove(body)

        elif mode == 1:
            pass

        # Paint scene
        # Show labels
        w.create_text(window_w/2, 40, text=mode_text[mode], fill='white', font='Avenir 30')
        w.create_text(window_w/2, window_h-50, text=mode_subtext[mode], fill='gray', font='Avenir 20')
        w.create_text(window_w / 2, 80, text=instructions[mode], fill='gray', font='Avenir 20')
        # Mode specific painting
        if mode == 0:
            for body in bodies:
                color = 'red'
                if 1 <= body.solver_id <= 2:
                    color = 'blue'
                elif body.solver_id >= 3:
                    color = 'green'
                w.create_text(*(A(body.xy) % [window_w, window_h] - np.array([0, 60])), text=str(body.solver_id), fill=color, font='Avenir 25')
                w.create_polygon(*body.get_poly_points(), fill=color, outline='white')
                w.create_oval(body.get_centroid_points(), fill='blue')

            if gravity:
                if frame < max_frame:
                    w.create_text(window_w / 2, window_h / 2, text='Competition Mode: Frame #'+str(frame) + '/'+str(max_frame), font='Avenir 50', fill='white')
                elif frame < max_frame + 100:
                    winner_text = '1st Place — S.I. Methods (Greens)!'
                    w.create_text(window_w/2, window_h/2, text=winner_text, font='Avenir 50', fill='white')
            else:
                w.create_text(window_w / 2, window_h / 2, text='Space mode (constant velocity, so all methods match exactly!)', font='Avenir 50', fill='white')

        elif mode == 1:
            pass

        # End run
        w.update()
        time.sleep(dt)


# Key bind
def key_pressed(event):
    global gravity, bodies, mode
    if event.char == 'm':
        mode = (mode + 1) % 2

    if mode == 0:
        if event.char == 'g':
            bodies.clear()
            gravity = not gravity

        # else:
        #     run()


# MOUSE + KEY METHODS ————————————————————————————————————————————————
def mouse_click(event):
    global mode, gravity, frame, bodies
    clicked_pt = Ainv(np.array([event.x, event.y]))
    if mode == 0 and ((len(bodies) == 0 and gravity) or (not gravity)):
        frame = 0
        # Sample random initial velocities and shape
        sidelen = 50
        rng_v = (np.random.random(2) - 0.5) * 300.0     # arbitrary max mag
        rng_theta = np.random.random() * 2 * np.pi      # arbitrary max mag
        rng_omega = (np.random.random() - 0.5) * 5.0    # arbitrary max mag
        rng_sides = np.random.randint(3, 11)            # arbitrary range
        verts = regular_ngon_verts(rng_sides, sidelen=sidelen) #+ np.random
        # Create four duplicates of this sample in a line, but each has a different integration scheme
        spacing = np.array([20.0 + (sidelen / np.sin(np.pi / rng_sides)), 0])  # from centroid to centroid. formula is for diameter of a /regular/ poly (good approx) plus some offset
        for i in range(5):
            xy = clicked_pt + (-2 * spacing) + (i * spacing)
            body = RigidBody(verts, init_xy=copy.copy(xy), init_v=copy.copy(rng_v), init_theta=copy.copy(rng_theta), init_omega=copy.copy(rng_omega), solver_id=i)
            bodies.append(body)


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