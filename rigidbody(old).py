import numpy as np
from tkinter import *
import time
import random
import copy

# Default Window Size
n = 10
window_w = int(2**n)
window_h = int(2**n)

# Tkinter Setup
root = Tk()
root.title("Rigid Body")
root.attributes("-topmost", True)
root.geometry(str(window_w) + "x" + str(window_h))  # window size hardcoded
w = Canvas(root, width=window_w, height=window_h)
w.pack()


# Coordinate Shift
def A(x, y):
    return np.array([x + window_w/2, -y + window_h/2])


# Helper functions
def rotmat(angle):
    # Angle must be in radians
    return np.column_stack([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])

class RigidBody:
    def __init__(self, points, init_xy=np.array([0.,0.]), init_v=np.array([0.,0.]), init_theta=0., init_omega=0., mass=1., inertia=500., gravity=0., id='none'):
        '''
            points: A python list of n np_arrays of size 2 comprising of the vertices forming the body,
                    in counter-clockwise order (IN ITS OWN REFERENCE FRAME).
            init_x: An np_array of size 2 denoting initial position.
            init_v: An np_array of size 2 denoting initial velocity.
            init_theta: A double denoting initial rotation angle.
            init_omega: A double denoting initial angular velocity.
        '''

        self.points = [np.array([p[0], p[1]]) for p in points]
        self.centroid = self.compute_centroid()
        self.xy, self.v, self.theta, self.omega = init_xy, init_v, init_theta, init_omega
        self.xy_stored, self.v_stored, self.theta_stored, self.omega_stored = None, None, None, None
        self.accel, self.alpha = np.array([0., gravity]), 0.
        self.m = mass
        self.I = inertia
        self.ID = id

        self.collision_buddy = None
        self.collision_point = None

    def euler_update(self, dt, store=False):
        if store:
            self.xy_stored, self.v_stored = np.copy(self.xy), np.copy(self.v)
            self.theta_stored, self.omega_stored = np.copy(self.theta), np.copy(self.omega)

        self.v += self.accel * dt
        self.xy += self.v * dt
        self.omega += self.alpha * dt
        self.theta += self.omega * dt

    def revert(self):
        self.xy, self.v = np.copy(self.xy_stored), np.copy(self.v_stored)
        self.theta, self.omega = np.copy(self.theta_stored), np.copy(self.omega_stored)

    def reset_collision_data(self):
        self.collision_buddy = None
        self.collision_point = None

    # Assume convex poly, split into triangles and take weighted avg of COM
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
            loc = np.dot(rotmat(self.theta), p - self.centroid) + self.centroid + self.xy
            if screen_space:
                points.extend(A(*loc))
            else:
                points.extend(loc)

        return points

    def get_centroid_points(self, radius=5):
        return [*A(self.centroid[0] - radius + self.xy[0], self.centroid[1] - radius + self.xy[1]),
                *A(self.centroid[0] + radius + self.xy[0], self.centroid[1] + radius + self.xy[1])]


# Create regular ngon rigid bodies
def regular_ngon_verts(n_sides, sidelen=50):
    points = []
    for theta in np.arange(0, 2*np.pi, 2*np.pi / n_sides):
        p = (sidelen * np.cos(theta), sidelen * np.sin(theta))
        points.append(p)

    return points


# Collision Detection Methods
# NOTE: Will not detect if intersection has no corners overlapping!
def do_intersect(body1, body2):
    # Point in polygon raycasting method.
    pts1, pts2 = body1.get_poly_points(screen_space=False), body2.get_poly_points(screen_space=False)
    pts1, pts2 = np.reshape(pts1, (int(len(pts1) / 2), 2)), np.reshape(pts2, (int(len(pts2) / 2), 2))
    ray_radius = 100000

    # Check, for every vertex of either body, whether it is inside the other body.
    for k in range(2):
        for v in pts1:
            count = 0
            vfar = ((np.random.random(2) * 2) - 1) * ray_radius
            for i in range(len(pts2)):
                p1, p2 = pts2[i], pts2[(i+1) % len(pts2)]

                # Solve system if possible
                M = np.column_stack((p2 - p1, v - vfar))
                b = v - p1
                if np.linalg.det(M) != 0:
                    t = np.matmul(np.linalg.inv(M), b)
                    if 0 <= t[0] <= 1 and 0 <= t[1] <= 1:
                        count += 1

            if count % 2 == 1:
                if k == 0:
                    body1.collision_buddy = body2
                    body2.collision_buddy = body1
                    body1.collision_point = v
                else:
                    body2.collision_buddy = body1
                    body1.collision_buddy = body2
                    body2.collision_point = v

                return True

        pts2, pts1 = pts1, pts2

    return False


# Find simulation time steps dt per body
def get_dts(bodies, init_dt, n_iters=5):
    # Every body stores its own Euler dt variable, initialized at given dt.
    dts = np.ones(len(bodies)) * init_dt

    # For each body, we calculate the dt. For any pair of bodies that we detect a collision, we use bisection search
    # to find the collision time and set BOTH their dts accordingly.
    for i, body in enumerate(bodies):
        # Iterate through rest of bodies (that this one hasn't already been compared with) and calculate smallest
        # dt for which it will just barely skim the first object it is bound to collide by the current Euler sim.
        for j in range(i + 1, len(bodies)):
            body2 = bodies[j]

            k = 0
            while k < n_iters:
                # Simulate next Euler time step for both bodies using their dts currently stored in table.
                # Store old values only if k=0.
                body.euler_update(dts[i], store=k==0)
                body2.euler_update(dts[j], store=k==0)
                # print(k, dts[1], body2.xy)

                # If they're (still) going to collide, update both their dt values accordingly
                if do_intersect(body, body2):
                    # print('we DO intersect')
                    dts[i] -= dts[i] / 2
                    dts[j] -= dts[j] / 2

                else:
                    # If they're not gonna collide but k=0, just reset and we're good to go!
                    if k == 0:
                        bodies[i].revert()
                        bodies[j].revert()
                        break

                    # If they're not gonna collide, but we're inside the bisection search, update dts accordingly
                    # print('we DONT intersect')
                    dts[i] += dts[i] / 2
                    dts[j] += dts[j] / 2

                # Revert bodies and prepare for next iteration
                bodies[i].revert()
                bodies[j].revert()
                k += 1

    return dts


def collision_response(b_inside, b_outside, epsilon=0.7):
    '''
        b_inside: Body whose vertex is touching the edge of the other one
        b_outside: The other body
        point: The collision point (i.e. the vertex)
        epsilon: Coefficient of restitution (within range [0,1] where 1 means perfectly elastic collision)
    '''
    pts = b_outside.get_poly_points(screen_space=False)
    pts = np.reshape(pts, (int(len(pts) / 2), 2))
    point = b_inside.collision_point

    # Find which edge this vertex is closest to
    min_dist = 10000
    min_idx = 0
    for i in range(len(pts)):
        v1, v2 = pts[i], pts[(i+1) % len(pts)]
        hyp_len, base_len = np.linalg.norm(point - v1), np.linalg.norm(v2 - v1)
        dist = hyp_len * np.sin(np.arccos(np.dot(point - v1, v2 - v1) / (hyp_len * base_len)))
        if dist < min_dist:
            min_dist, min_idx = dist, i

    # Calculate collision normal vector from edge
    n = np.dot(rotmat(-np.pi/2), pts[(min_idx+1) % len(pts)] - pts[min_idx])
    n = np.array([n[0], n[1], 0]); n /= np.linalg.norm(n)

    # Compute impulse and new linear & angular velocities
    centroid = b_inside.centroid + b_inside.xy
    omega = np.array([0., 0., b_inside.omega])
    ra = np.array([(point - centroid)[0], (point - centroid)[1], 0])
    va_bar = b_inside.v + np.cross(omega, ra)[:2]

    centroid = b_outside.centroid + b_outside.xy
    omega[2] = b_outside.omega
    rb = np.array([(point - centroid)[0], (point - centroid)[1], 0])
    vb_bar = b_outside.v + np.cross(omega, rb)[:2]

    vab_bar = np.array([(va_bar - vb_bar)[0], (va_bar - vb_bar)[1], 0.])

    j = (-(1. + epsilon) * np.dot(vab_bar, n))
    j /= (np.dot(n, n) * ((1. / b_inside.m) + (1. / b_outside.m)))\
         + np.dot(n, np.cross(np.cross(ra, n) / b_inside.I, ra) + np.cross(np.cross(rb, n) / b_outside.I, rb))

    b_inside.v += (j / b_inside.m) * n[:2]
    b_outside.v -= (j / b_outside.m) * n[:2]
    b_inside.omega += np.cross(ra, j * n)[2] / b_inside.I
    b_outside.omega -= np.cross(rb, j * n)[2] / b_outside.I
    # print(rb, n, np.arccos(np.dot(rb, n) / (np.linalg.norm(rb) * np.linalg.norm(n))), np.cross(rb, j * n))
    # print()


# Main update method
def update(bodies, dt):
    dts = get_dts(bodies, dt, n_iters=10)

    for i, body in enumerate(bodies):
        body.euler_update(dts[i])
        buddy = body.collision_buddy
        if buddy is not None:
            if body.collision_point is not None:
                collision_response(body, buddy)
            elif buddy.collision_point is not None:   # to fix random error... don't know exactly why because I'm lazy
                collision_response(buddy, body)

            body.reset_collision_data()
            buddy.reset_collision_data()


# Create rigid bodies
# Remember that position matters based on where you place the local frame origin at wrt the shape geom.
# l, h = 200, 200
g = 10
# l, h = 20, 20
# rect1 = RigidBody([(-l/2, h/2), (-l/2, -h/2), (l/2, -h/2), (l/2, h/2)], init_theta=0., init_omega=np.pi/6, init_v=np.array([0., 100.]), gravity=g, id='rect1')
# rect2 = RigidBody([(-l/2, h/2), (-l/2, -h/2), (l/2, -h/2), (l/2, h/2)], init_theta=0., init_omega=0., init_v=np.array([0., -100.]), init_xy=np.array([100., 400.]), gravity=g, id='rect2')
# rect3 = RigidBody([(-l/2, h/2), (-l/2, -h/2), (l/2, -h/2), (l/2, h/2)], init_theta=0., init_omega=0., init_v=np.array([0., -100.]), init_xy=np.array([-200., 400.]), gravity=g, id='rect3')
# rect4 = RigidBody([(-l/2, h/2), (-l/2, -h/2), (l/2, -h/2), (l/2, h/2)], init_theta=0., init_omega=0., init_v=np.array([0., 100.]), init_xy=np.array([0., -300.]), gravity=g, id='rect4')
# rigid_bodies = [rect1, rect2, rect3, rect4]

# l, h = 30, 200
# rect = RigidBody([(-l/2, h/2), (-l/2, -h/2), (l/2, -h/2), (l/2, h/2)], init_theta=np.pi/2., init_omega=0., init_v=np.array([0., 0.]))
# tri1 = RigidBody(regular_ngon_verts(3), init_xy=np.array([-70., 100.]), init_v=np.array([0., -100.]), init_omega=0., init_theta=np.pi/6)
# tri2 = RigidBody(regular_ngon_verts(3), init_xy=np.array([70., 100.]), init_v=np.array([0., -100.]), init_omega=0., init_theta=np.pi/6)
#
# # Add all bodies to list of bodies to simulate
# rigid_bodies = [rect, tri1, tri2]

# Cool configuration that works:
l, h = 60, 200
tri = RigidBody(regular_ngon_verts(3), init_xy=np.array([-80., -150.]), init_v=np.array([0., 100.]), init_omega=-1., gravity=g)
rect = RigidBody([(-l/2, h/2), (-l/2, -h/2), (l/2, -h/2), (l/2, h/2)], init_theta=np.pi/2, init_omega=1., init_v=np.array([-10., 0.]), gravity=g)
h=l
rect2 = RigidBody([(-l/2 * 3, h/2), (-l/2, -h/2), (l/2, -h/2), (l/2, h/2)], init_xy=np.array([-300., 30.]), init_theta=0., init_omega=-2., init_v=np.array([300., 100.]), gravity=g)
pent = RigidBody(regular_ngon_verts(5, sidelen=100), init_xy=np.array([200., 300.]), init_v=np.array([-200., -100.]), init_omega=-1., gravity=g, mass=20.)
l, h = window_w, 20
ground = RigidBody([(-l/2, h/2), (-l/2, -h/2), (l/2, -h/2), (l/2, h/2)], init_theta=0., init_omega=0., init_v=np.array([0., 0.]), init_xy=np.array([0., -window_w/2. + 40.]), id='ground', mass=20000., inertia=200000000.)
rigid_bodies = [tri, rect, rect2, pent]
#
# # Other params
first_run = True
show_centroids = True
dt = 0.01


# Main function
def run():
    global first_run, dt, rigid_bodies
    w.configure(background='black')

    # Update Model
    update(rigid_bodies, dt)

    # Paint scene
    if first_run:
        for i, body in enumerate(rigid_bodies):
            w.create_polygon(body.get_poly_points(),  fill='red', outline='white', tags='body'+str(i))
            if show_centroids:
                w.create_oval(body.get_centroid_points(), fill='blue', tags='com'+str(i))

    else:
        for i, body in enumerate(rigid_bodies):
            poly = w.find_withtag('body'+str(i))
            w.coords(poly, body.get_poly_points())
            if show_centroids:
                circle = w.find_withtag('com'+str(i))
                w.coords(circle, body.get_centroid_points())

    # End run
    first_run = False
    w.update()
    time.sleep(dt)


# Main function
if __name__ == '__main__':
    while True:
        run()

# Necessary line for Tkinter
mainloop()