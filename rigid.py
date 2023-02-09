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
    def __init__(self, points, init_xy=np.array([0.,0.]), init_v=np.array([0.,0.]), init_theta=0., init_omega=0.,
                 mass=1., inertia=500., solver_id=0, id='none'):
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

        self.bbox = self.compute_bbox()

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

    # Compute the top-left and bottom-right of bbox
    # IN THE SHAPE'S REFERENCE FRAME! (In the usual space of (0, 0) being at the center of the screen).
    def compute_bbox(self, fluff=40):
        large = 1000000
        xy_high = np.array([-large, -large])
        xy_low = np.array([large, large])
        for p in self.points:
            xy_high = np.maximum(xy_high, p)
            xy_low = np.minimum(xy_low, p)

        return np.array([xy_low[0] - fluff, xy_high[1] + fluff]), np.array([xy_high[0] + fluff, xy_low[1] - fluff])

    def world_coords(self, local_vec, is_nor=False):
        '''
            Converts local shape space vector (e.g. a vertex location in shape space) into world.
            (Still not in screen space though, i.e. NOT applying A() function.)

            is_nor: Normals transform differently. If this is true, we apply that transformation.
        '''
        p = copy.copy(local_vec)
        if not is_nor:
            return np.dot(R(self.theta), p - self.centroid) + self.centroid + self.xy

        return np.dot(R(self.theta), p)   # TODO: NOT TRUE IF IT'S NOT A REGULAR POLY AND HENCE CENTROID IS \NEQ 0

    def shape_coords(self, world_vec, is_nor=False):
        p = copy.copy(world_vec)
        if not is_nor:
            return np.dot(R(self.theta).T, p - self.xy - self.centroid) + self.centroid

        return np.dot(R(self.theta).T, p)

    def get_poly_points(self, screen_space=True):
        # Assume x, theta have already been updated
        points = []
        for p in self.points:
            # Rotate about center of mass
            #xy = Ainv(A(self.xy) % [window_w, window_h])
            loc = self.world_coords(p)
            if screen_space:
                points.extend(A(loc))
            else:
                points.extend(loc)

        return points

    def get_centroid_points(self, radius=5):
        xy = self.xy #Ainv(A(self.xy) % [window_w, window_h])
        px, py = self.centroid - radius + xy, self.centroid + radius + xy
        return [*A(px), *A(py)]

    def get_bbox_points(self):
        pts = []
        for p in self.bbox:
            pts.extend(A(p + self.xy))
        return pts

    # NOT in screenspace NOR worldspace! (i.e. NOT upon application of A())
    def get_endpoint_val(self, kind):
        '''
            kind: 0 = x-start, 1=x-end, 2=y-start, 3=y-end
        '''
        topleft, botright = copy.copy(self.bbox[0]), copy.copy(self.bbox[1])
        topleft += self.xy
        botright += self.xy
        if kind == 0:
            return topleft[0]
        elif kind == 1:
            return botright[0]
        elif kind == 2:
            return botright[1]

        return topleft[1]

    def KE(self):
        return 0.5 * np.dot(self.v, self.v) * self.m

    def solve(self, h, const_a=np.array([0,0]), const_alpha=0., experiment=0):
        params = [h, const_a, const_alpha]
        if experiment == 0:
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

        elif experiment > 0:
            if self.solver_id == 0:
                self.solve_forward_euler(*params)
            elif self.solver_id == 1:
                self.solve_RK2(*params)
            elif self.solver_id == 2:
                self.solve_verlet(*params)
            elif self.solver_id == 3:
                self.solve_sieuler_expanded(*params)  # naive will give equivalent results
            elif self.solver_id != -1:
                self.solve_backward_euler(h)

    # BELOW LIE ALL THE IMPLICIT INTEGRATORS (THEY REQUIRE SOME FORM OF MATRIX INVERSION)!
    # —————————————————————————————————————————————————————————————————————————————————————————————
    def solve_backward_euler(self, h):
        dfdx = vectorfield_jacobian(self.xy)
        A = np.eye(2) - (np.power(h, 2.0) * dfdx)
        b = vectorfield(self.xy) + (h * np.dot(dfdx, self.v))
        self.v += h * np.dot(np.linalg.inv(A), b)
        self.xy += h * self.v


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

    # Just a wrapper method! (To make clear Trapezoid rule is literally just the below.)
    def solve_RK2(self, h, a_n, alpha_n):
        '''
            Here, a_n and alpha_n are the PREVIOUS time-step's accelerations as this is an EXPLICIT method.
        '''
        return self.solve_exact_tk(h, a_n, alpha_n)

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

    # 6) Verlet integration (velocity verlet, as we'll store the velocities too)
    #    Recall that normal Verlet integration does not compute velocity.
    def solve_verlet(self, h, a_n, alpha_n):
        # NOTE: This is for the specific vectorfield case (does not affect rotational component)
        h_squared = np.power(h, 2.0)
        self.xy += (h * self.v) + (0.5 * a_n * h_squared)
        a_np1 = vectorfield(self.xy)
        self.v += 0.5 * (a_np1 + a_n) * h
        self.theta += (h * self.omega) + (0.5 * alpha_n * h_squared)
        self.omega += alpha_n * h  # constant angular acceleration means 0.5 * (alpha_{n+1} + alpha_n) = alpha_n.

    def __repr__(self):
        return str(self.ID)


# Get regular n-gon vertices
def regular_ngon_verts(n_sides, sidelen=50):
    points = []
    for theta in np.arange(0, 2*np.pi, 2*np.pi / n_sides):
        p = (sidelen * np.cos(theta), sidelen * np.sin(theta))
        points.append(p)

    return points

# Normalize a vector
def normalize(vec: np.ndarray):
    return vec / np.linalg.norm(vec)


# Switches / Knobs 🎚
mode = 2   # 0 = Unconstrained rigid bodies (no forces), 1 = N-body comparison, 2 = Rigid body collision

# Bells and whistles 🔔
mode_text = ['EXPERIMENT #1: Constant Velocity / Acceleration Integration',
             'EXPERIMENT #2: Non-constant Force Integration',
             'EXPERIMENT #3: Rigid Body Collisions']

mode_subtext = ['Constant Velocity: All will tie. Constant Acceleration: Semi-implicit gains extra energy and wins, while Euler strays behind the actual solution!',
                '1) Red = Forward Euler, 2) Blue = RK2, 3) Green = Verlet (Velocity Version), 4) Purple = SI Euler, 5) Orange = Implicit Euler (Pixar\'s Notes)',
                'Objects constantly tunnel through each other, and even if we were to use bisection search to find exact c.p., not clear how large collision sims would stop tunneling.']

instructions = ['Click anywhere to spawn 5 rigid bodies, one per integrator. \'G\' to toggle between space-mode and competition-mode.',
                'Instructions: Sit back and watch.',
                'Overall: Impulse Based Local Method — Very Messy :(']

# Universal parameters 🌎
bodies = []
gravity = True

# Simulation params
dt = 1. / 60.  # 60 fps

# Mode 0 parameters / functions
frame = 0
max_frame = 400


# Mode 1 parameters / functions
energies = []
initial_energies = []
const = 100000000
epsilon = 10
def vectorfield(xy: np.ndarray):
    global const, epsilon
    x, y = xy[0], xy[1]
    r_mag = np.linalg.norm(xy)
    factor = -const / np.power(r_mag, 3.0) if r_mag > epsilon else 0.0
    return np.array(factor * np.array([x, y]))


# Jacobian of vectorfield wrt to the input position
def vectorfield_jacobian(xy: np.ndarray):
    global const, epsilon
    x, y, r_mag = xy[0], xy[1], np.linalg.norm(xy)
    if r_mag <= epsilon:
        return np.zeros((2,2))

    rpower_3 = np.power(r_mag, -3.0)
    rpower_5 = np.power(r_mag, -5.0)
    Jrow0 = np.array([(const * rpower_3) - (3 * const * np.power(x, 2.0) * rpower_5), -3 * const * x * y * rpower_5])
    Jrow1 = np.array([-3 * const * x * y * rpower_5, (const * rpower_3) - (3 * const * np.power(y, 2.0) * rpower_5)])
    J = np.array([Jrow0, Jrow1])
    return J


def vectorfield_potential(xy: np.ndarray):
    global const, epsilon
    r_mag = np.linalg.norm(xy)
    return - const / r_mag if r_mag > epsilon else 0


# Mode 2 Parameters
restitution_coeff = 1.0
class Endpoint:
    kinds = ['X-Start', 'X-End', 'Y-Start', 'Y-End']

    def __init__(self, body_idx, value, kind=0):
        '''
            Simple class which stores either the start or end of the bbox of an object for a certain dimension.
            body_idx: Index of the body in question wrt the global 'bodies' array.
            value: The actual value of the endpoint in question. NEEDS TO BE REFRESHED CONSTANTLY!
            kind: 0 = x-start, 1=x-end, 2=y-start, 3=y-end (we only really need 2 types (don't care which dimension,
                                                            but for debugging purposes))
        '''
        self.idx = body_idx
        self.val = value
        self.kind = kind

    def refresh_val(self):
        global bodies
        self.val = bodies[self.idx].get_endpoint_val(kind=self.kind)

    def __lt__(self, other):
        return self.val < other.val

    def __eq__(self, other):
        return self.val == other.val

    def __repr__(self):
        return str(Endpoint.kinds[self.kind]) + ': ' + str(self.val)


x_sorted_endpoints = []
y_sorted_endpoints = []
def get_collision_pairs():
    global x_sorted_endpoints, y_sorted_endpoints
    # Use sort & sweep algorithm
    # 1. Sort (will be fast as it'll be already nearly sorted in general)
    x_sorted_endpoints = sorted(x_sorted_endpoints)
    y_sorted_endpoints = sorted(y_sorted_endpoints)
    # 2. Sweep
    x_pairs = set()
    active_idxes = []
    # In x-dimension first
    for endpoint in x_sorted_endpoints:
        if endpoint.kind == 0:
            for idx in active_idxes:
                x_pairs.add(frozenset({idx, endpoint.idx}))
            active_idxes.append(endpoint.idx)
        else:
            active_idxes.remove(endpoint.idx)
    # Now y-dimension
    y_pairs = set()
    active_idxes.clear()
    for endpoint in y_sorted_endpoints:
        if endpoint.kind == 2:
            for idx in active_idxes:
                y_pairs.add(frozenset({idx, endpoint.idx}))
            active_idxes.append(endpoint.idx)
        else:
            active_idxes.remove(endpoint.idx)

    return list(x_pairs & y_pairs)

# Dict from a 2-FROZENSET representing a collision pair, to a 2-tuple / 2-list representing the plane.
# The plane is represented as (r0, w, i) where r0 and v are np.arrays representing the offset and normal, respectively,
# IN THE SHAPE'S LOCAL SPACE! Otherwise, it's no use! For every rotational / positional change we have to recompute,
# and i is the index of the body whose local frame is the one we're using to describe r0 and w.
# i.e. The test is then whether f(x) = w • (x - r0) is positive or negative.
# NOTE: We'll NEVER remove pairs from this dict. It doesn't matter. If a pair is not a valid bbox collision, we just
#       won't access it, and some dirty data will remain its value, which'll be cleaned automatically by checking
#       if it's a valid plane if the pairs' bboxes happen to intersect again in the future.
separating_planes = {}
plane_points = []  # purely for visualization
collision_points = []  # for viz
def valid_plane(pair: frozenset, r0: np.ndarray, nor: np.ndarray, get_degenerates=False):
    '''
        Assumptions:
        1. The shape of the body is convex.
        2. We'll say a separating plane satisfies:
            — ALL the verts of one of the bodies must result in f(v) > 0  (strict positivity), AND
            — ALL the verts of the other must result in f(v) <= 0  (weak negativity).


        get_degenerates: Boolean, if True then if there the plane is not valid, we get a list of vertices that
                         fall on the "wrong side" of the plane, e.g. in a vertex-edge collision, this will be
                         the single vertex constituting the collision point.

        NOTE: THE INPUT r0 AND nor HAVE TO BE IN WORLD SPACE!!
    '''
    global bodies
    b1, b2 = bodies[list(pair)[0]], bodies[list(pair)[1]]
    eps = 1E-10
    degens = []

    raw_pts = b1.get_poly_points(screen_space=False)
    pts = np.reshape(raw_pts, (int(len(raw_pts) / 2), 2))
    correct_side = np.dot(nor, pts[0] - r0) > eps
    for vert in pts:
        same_side = correct_side == (np.dot(nor, vert - r0) > eps)
        if not same_side:
            degens.append(vert)
            if not get_degenerates:
                return False

    # Fact that we made it here means one of the shapes falls entirely on one side of the plane.
    # Now we needa check the other one.
    raw_pts = b2.get_poly_points(screen_space=False)
    pts = np.reshape(raw_pts, (int(len(raw_pts) / 2), 2))
    for vert in pts:
        opp_side = correct_side != (np.dot(nor, vert - r0) > eps)
        if not opp_side:
            degens.append(vert)
            if not get_degenerates:
                return False

    # Return stuff
    if not get_degenerates:
        return True
    else:
        return degens


def find_plane(pair: frozenset):
    '''
        Basically handle everything having to do with separating planes.
        1) If the pair already has an associated plane, query it and check if still valid.
            a) If valid, then return it.
            b) Otherwise, proceed.
        2) Search for a new separating plane (exhaustively).
            a) If found: (i) Add it to the dictionary, and (ii) Return it.
            b) If not found, there's been contact. So return None.
    '''
    global separating_planes, bodies
    if pair in separating_planes.keys():
        r0, nor, idx = separating_planes[pair]
        r0 = bodies[idx].world_coords(r0)
        nor = bodies[idx].world_coords(nor, is_nor=True)
        if valid_plane(pair, r0, nor):  # bodies[idx].world_coords(nor, is_nor=True)
            # print('Found cached!')
            return r0, nor

    # Either it's a new pair we've never encountered, or old separating plane no longer valid.
    # print('Finding new one...')
    lst_pair = list(pair)
    for k in range(2):
        body = bodies[lst_pair[k]]
        pts = body.get_poly_points(screen_space=False)
        n = int(len(pts) / 2)
        pts = np.reshape(pts, (n, 2))
        for i in range(n):
            edge = (pts[(i+1) % n] - pts[i]); edge /= np.linalg.norm(edge)
            nor = np.array([edge[1], -edge[0]])
            r0 = pts[i]  # arbitrary choice among ith and (i+1)th point
            if valid_plane(pair, r0, nor):
                # print('Stored new one')
                # Encode + store (Below command works regardless of whether pair already exists in dict)
                separating_planes[pair] = (body.shape_coords(r0), body.shape_coords(nor, is_nor=True), lst_pair[k]) #(r0, nor, lst_pair[k])
                return r0, nor

    # Fact that we're here means no separating plane (neither cached nor new) exists. There's been contact.
    # print('Could not find one.')
    return None


# Main function
def run():
    global dt, frame, max_frame, mode, bodies, gravity, mode_subtext, energies, initial_energies, \
           x_sorted_endpoints, y_sorted_endpoints, separating_planes, plane_points, collision_points, \
           restitution_coeff
    w.configure(background='black')

    # Prep
    if mode == 0:
        pass

    elif mode == 1:
        # Create all the ghosts of the same body, but being integrated using different methods
        n_bodies = 5     # 1) FW Euler, 2) RK2, 3) Verlet, 4) SI Euler, 5) Implicit Euler (from Pixar's notes)
        for i in range(n_bodies):
            xy0 = np.array([-300.0, 0.0])
            v0 = np.array([0.0, 500.0])
            verts = regular_ngon_verts(5)
            body = RigidBody(verts, init_xy=xy0, init_v=v0, solver_id=i)
            bodies.append(body)

            # Store initial total energies
            energy = (body.KE() / body.m) + vectorfield_potential(body.xy)
            energies.append(energy)

        initial_energies = copy.copy(energies)

    elif mode == 2:
        # Spawn 3 bodies heading towards each other, and sort them by bbox
        center = np.array([0, 0])
        radius = 200
        n_bodies = 4
        for i in range(n_bodies):
            # Create the rigid body
            theta = 2.0 * np.pi * float(i) / n_bodies
            xy0 = radius * np.array([np.cos(theta), np.sin(theta)])
            v0 = (normalize(center - xy0) * 200.0) + np.array([0, 600 if gravity else 0])
            verts = regular_ngon_verts(3)
            body = RigidBody(verts, init_xy=xy0, init_v=v0, init_theta=0, init_omega=np.pi/5 * (i+1), solver_id=1)
            bodies.append(body)
            # Add its bbox endpoints to the lists (currently will be unsorted)
            x_sorted_endpoints.extend([Endpoint(i, body.get_endpoint_val(0), 0),
                                       Endpoint(i, body.get_endpoint_val(0), 1)])
            y_sorted_endpoints.extend([Endpoint(i, body.get_endpoint_val(2), 2),
                                       Endpoint(i, body.get_endpoint_val(3), 3)])

        # Add ground and walls to the list of bodies
        # 1. Ground
        eps = 40
        ground_verts = [[-window_w/2, eps/2], [-window_w/2, -eps/2], [window_w/2, -eps/2], [window_w/2, eps/2]]
        ground = RigidBody(np.array(ground_verts), init_xy=np.array([0, -window_h/2]), mass=1E20, inertia=1E20, solver_id=-1)
        bodies.append(ground)
        x_sorted_endpoints.extend([Endpoint(n_bodies, ground.get_endpoint_val(0), 0),
                                   Endpoint(n_bodies, ground.get_endpoint_val(0), 1)])
        y_sorted_endpoints.extend([Endpoint(n_bodies, ground.get_endpoint_val(2), 2),
                                   Endpoint(n_bodies, ground.get_endpoint_val(3), 3)])

        # # 2. Left wall
        # left_verts = [[-window_w / 2, eps / 2], [-window_w / 2, -eps / 2], [window_w / 2, -eps / 2], [window_w / 2, eps / 2]]
        # left_wall = RigidBody(np.array(left_verts), init_xy=np.array([-window_w/2, 0]), init_theta=np.pi/2, mass=1E20, inertia=1E20, solver_id=-1, id='left')
        # bodies.append(left_wall)
        # x_sorted_endpoints.extend([Endpoint(n_bodies+1, left_wall.get_endpoint_val(0), 0),
        #                            Endpoint(n_bodies+1, left_wall.get_endpoint_val(0), 1)])
        # y_sorted_endpoints.extend([Endpoint(n_bodies+1, left_wall.get_endpoint_val(2), 2),
        #                            Endpoint(n_bodies+1, left_wall.get_endpoint_val(3), 3)])

        # # 3. Right wall
        # right_verts = [[-window_w / 2, eps / 2], [-window_w / 2, -eps / 2], [window_w / 2, -eps / 2], [window_w / 2, eps / 2]]
        # right_wall = RigidBody(np.array(right_verts), init_xy=np.array([window_w/2, 0]), init_theta=np.pi/2, mass=1E20, inertia=1E20, solver_id=-1)
        # bodies.append(right_wall)
        # x_sorted_endpoints.extend([Endpoint(n_bodies + 2, right_wall.get_endpoint_val(0), 0),
        #                            Endpoint(n_bodies + 2, right_wall.get_endpoint_val(0), 1)])
        # y_sorted_endpoints.extend([Endpoint(n_bodies + 2, right_wall.get_endpoint_val(2), 2),
        #                            Endpoint(n_bodies + 2, right_wall.get_endpoint_val(3), 3)])
        #
        # # 4. Ceiling
        # ceil_verts = [[-window_w / 2, eps / 2], [-window_w / 2, -eps / 2], [window_w / 2, -eps / 2], [window_w / 2, eps / 2]]
        # ceiling = RigidBody(np.array(ceil_verts), init_xy=np.array([0, window_h / 2]), mass=1E20, inertia=1E20, solver_id=-1)
        # bodies.append(ceiling)
        # x_sorted_endpoints.extend([Endpoint(n_bodies + 3, ceiling.get_endpoint_val(0), 0),
        #                            Endpoint(n_bodies + 3, ceiling.get_endpoint_val(0), 1)])
        # y_sorted_endpoints.extend([Endpoint(n_bodies + 3, ceiling.get_endpoint_val(2), 2),
        #                            Endpoint(n_bodies + 3, ceiling.get_endpoint_val(3), 3)])


    bang = False
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
            for i, body in enumerate(bodies):
                accel = vectorfield(body.xy)
                body.solve(dt, const_a=accel, experiment=mode)
                energy = vectorfield_potential(body.xy) + (body.KE() / body.m)
                energies[i] = energy

        elif mode == 2:
            # TODO: Handle collision detection + resolution here (i.e. before solving ODEs)
            # TODO:
            #       1) Take an ODE step.
            #       2) Maintain sorted lists of bodies (1 for each dimension). Re-sort each time using insertion sort.
            #       3) Sweep through the sorted lists to find all bbox collisions, and store them in a list of pairs
            #          i.e. (body1, body2).
            #       4) Go through the list of pairs and either compute a new separating plane between the two shapes and
            #          store it (in a way that they're associated with the pair), or check if the already stored one
            #          is still valid. If there IS NO SEPARATING PLANE (but bboxes intersect), then contact!
            #          Question: What kind of contact is it?
            #                       - Compute contact normal (easy, just perpendicular to separating plane)
            #                       - Compute v_relative in direction of normal (normal • delta_v's)
            #                       - If v_rel < -eps  =>  colliding contact,  -eps < v_rel < eps  =>  resting,
            #                         v_rel > eps  =>  nothing to do. They're already moving apart.
            #               4a) If colliding contact, then use bisection search to find exact time of collision (solve
            #                   ODE from last time-step forward in time through various amounts to conduct the search).
            #                   Then compute proper response (update velocities immediately, but not positions).
            #               4b) If resting contact, there's a problem if there are external forces, because we might
            #                   still need movement (i.e. the configuration might not be stable). So gravity, say,
            #                   is pulling us down——BUT——if all the contact points are treated as collisions, since
            #                   velocity is so small, the impulse will be so tiny / non-existent that the collision
            #                   point remains unresolved for the next time-step, and the next, and the next, and so on.
            #                   So one way to really allow for the sliding motion which is what's really required is
            #                   to just exert the proper contact forces required to keep the object in place and not
            #                   penetrating. Then the vector sum of the forces (taking into account gravity) will be
            #                   tangential and create the necessary sliding motion.

            # 0) Take an ODE step.
            for b in bodies:
                b.solve(dt, const_a=np.array([0, -700 if gravity else 0]), experiment=mode)

            # 1) Refresh the endpoints in the lists to reflect correct value (for resorting)
            for x_pt, y_pt in zip(x_sorted_endpoints, y_sorted_endpoints):
                x_pt.refresh_val()
                y_pt.refresh_val()

            # 2) Iterate through pairs and find separating plane, if it exists (either from cache or find new)
            plane_points.clear()  # every iteration the points change (for drawing)
            collision_points.clear()
            for pair in get_collision_pairs():
                plane = find_plane(pair)
                # a) If there is a valid plane, sample it for drawing.
                if plane is not None:
                    r0, nor = plane
                    v = np.array([-nor[1], nor[0]]); v /= np.linalg.norm(v)
                    p1, p2 = r0 + (-200 * v), r0 + (+200 * v)
                    plane_points.append([*A(p1), *A(p2)])
                # b) Otherwise, we have contact and might need to handle it.
                else:
                    bang = True
                    # Get the most recently encountered plane for this pair (there must have been one, and
                    # it must be in the cache). And use it to compute the relative linear velocity wrt the normal.
                    r0, nor, idx = separating_planes[pair]
                    idx1, idx2 = tuple(pair)
                    if idx1 != idx:  # we want the relative velocity to be computed as := (other_obj's vel) – (sep_plane_obj's vel)
                        idx1, idx2 = idx2, idx1
                    nor = bodies[idx].world_coords(nor, is_nor=True)
                    r0 = bodies[idx].world_coords(r0)
                    collision_points.extend(valid_plane(pair, r0, nor, get_degenerates=True))

                    # Drawing purposes... (separating plane)
                    v = np.array([-nor[1], nor[0]]); v /= np.linalg.norm(v)
                    p1, p2 = r0 + (-200 * v), r0 + (+200 * v)
                    plane_points.append([*A(p1), *A(p2)])

                    # Classify the contact type
                    body_a, body_b = bodies[idx1], bodies[idx2]
                    pt = collision_points[0]
                    ra, rb = pt - body_a.world_coords(body_a.centroid), pt - body_b.world_coords(body_b.centroid)
                    ra, rb = np.array([*ra, 0]), np.array([*rb, 0])
                    va_bar = body_a.v + np.cross(np.array([0., 0., body_a.omega]), ra)[:-1]
                    vb_bar = body_b.v + np.cross(np.array([0., 0., body_b.omega]), rb)[:-1]
                    vab_bar = np.array([*(va_bar - vb_bar), 0])
                    nor = np.array([*nor, 0])
                    v_rel = np.dot(nor, -vab_bar)
                    eps = 1E-10
                    # 1. Collision Contact
                    if v_rel < -eps:
                        # if len(collision_points) == 1:
                            # print('Single Point')
                        # Impulse computation
                        num = -(1 + restitution_coeff) * np.dot(vab_bar, nor)
                        # # TODO: Actually compute the correct inertia scalar using Green's theorem.
                        denom = ((1.0 / body_a.m) + (1.0 / body_b.m) + np.dot(nor, (np.cross(np.cross(ra, nor) / body_a.I, ra)) + (np.cross(np.cross(rb, nor) / body_b.I, rb))))
                        j = num / denom

                        # Velocity / Angular Velocity Update
                        body_a.omega += (np.cross(ra, j * nor) / body_a.I)[-1]
                        body_b.omega -= (np.cross(rb, j * nor) / body_b.I)[-1]
                        nor = nor[:-1]
                        body_a.v += j * nor / body_a.m
                        body_b.v -= j * nor / body_b.m

                        # else:
                        #     print('Uh oh, multiple points. Not sure how to handle.')

                    # 2. Resting Contact
                    elif -eps < v_rel < eps:
                        pass # RESTING

                    # 3. Separating Contact
                    else:
                        print('Separating')
                        pass # Nothing to do, separating




            # print('Number of planes:', len(plane_points))
            # print('Number cached:', len(separating_planes.keys()))

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
            sink_radius = 10
            w.create_oval(window_w/2 - sink_radius, window_h/2 - sink_radius,
                          window_w/2 + sink_radius, window_h/2 + sink_radius, fill='white')

            colors = ['red', 'blue', 'green', 'purple', 'orange']
            for i, body in enumerate(bodies):
                col = colors[body.solver_id]
                w.create_polygon(*body.get_poly_points(), fill=col, outline='white')
                w.create_oval(body.get_centroid_points(), fill='blue')

                # Display energies
                width = 40
                topleft = A(np.array([-window_w * 0.8 / 2, 0.0]) + np.array([width * i, 0]))
                scaling = 1.0 / 1500.0
                energy = energies[i]
                w.create_rectangle(*topleft, *(topleft + np.array([width, energy * scaling])), fill=col)

                # Display lines for initial energies
                energy = initial_energies[i]
                w.create_line(*(topleft + np.array([0, energy * scaling])), *(topleft + np.array([width, energy * scaling])), fill=col)
                
        elif mode == 2:
            # Draw bodies
            for b in bodies:
                w.create_polygon(*b.get_poly_points(), fill='blue', outline='white')
                w.create_rectangle(*b.get_bbox_points(), fill='', outline='white')

            # Draw all separating planes
            for points in plane_points:
                w.create_line(*points, fill='orange', width=3)

            # Draw all collision points
            radius = 10
            for pt in collision_points:
                w.create_oval(*A(pt - radius), *A(pt + radius), fill='red')


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