import taichi as ti # pyright: ignore[reportMissingImports]
import taichi.math as tm # pyright: ignore[reportMissingImports]
ti.init(arch=ti.cpu, default_ip=ti.i32, default_fp=ti.f64)

# Screen size
width, height = 800, 800

# Type and Class declarations
vec2 = tm.vec2

@ti.dataclass
class Particle:
    i: int   # index in particle_field
    r: vec2  # type: ignore | position in cartesian coordinates
    v: vec2  # type: ignore | veloctiy
    h: float # smoothing length; desnisty (p) is directly related to h, so it can be calculated when needed with p_calc()
    m: float # mass
    u: float # internal energy
    O: float # Omega


# Constants
n = 1000 # Number of particles
sigma: float = 7 / (478 * tm.pi) # Normalization constant for M6 in 2D

# The actual particle field
particle_field = Particle.field(shape=(n,))

# A dynamic list to store neighbours of particles temporarily
S = ti.root.dynamic(ti.i, 1024, chunk_size=32)
dynamic_list = ti.field(dtype=int)
S.place(dynamic_list)

S2 = ti.root.dynamic(ti.i, 1024, chunk_size=32)
dynamic_list2 = ti.field(dtype=int)
S2.place(dynamic_list2)

"""
Helper functions
"""

@ti.func
def p_calc(m: float, h: float, h_fact: float) -> float:
    return m * (h_fact/h)**2

@ti.func
def P_calc(p: float, u: float, gamma: float):
    # Ideal gas equation of state, gamma is the adiabatic index
    return (gamma - 1) * p * u

@ti.func
def cs_calc(gamma: float, P: float, p: float):
    return tm.sqrt(gamma * P / p)

@ti.func
def r_hat_calc(r_a: float, r_b: float):
    return tm.normalize(r_a - r_b)

@ti.func
def v_sig_calc(r_a: float, r_b: float, cs_a: float, cs_b: float, v_a: vec2, v_b: vec2, beta: float): # type: ignore
    v_ab_dot_r_hat = tm.dot(v_a - v_b, r_hat_calc(r_a, r_b))
    result = 0.0
    if v_ab_dot_r_hat <= 0:
        result = 0.5 * (cs_a + cs_b - (beta * v_ab_dot_r_hat))
    return result

@ti.func
def v_sig_u_calc(P_a: float, P_b: float, p_a: float, p_b: float):
    return tm.sqrt(ti.abs(P_a - P_b)/((p_a + p_b)/2))


@ti.func
def M6(q: float) -> float:
    """M6 quintinc kernel smoothing in 2D, returns -1 if q < 0"""
    result = -1.0
    
    a = (3 - q)**5
    b = -6 * (2 - q)**5
    c = 15 * (1 - q)**5

    if q >= 0:
        result = a + b + c
    if q > 1:
        result = a + b
    if q > 2:
        result = a
    if q >= 3:
        result = 0
        
    return result

@ti.func
def M6_dq(q: float) -> float:
    """Derivative of M6 wrt q, returns -1 if q < 0"""
    result = -1.0
    
    a = -5 * (3 - q)**4
    b = 30 * (2 - q)**4
    c = -75 * (1 - q)**4

    if q > 0: 
        result = a + b + c
    if q > 1:
        result = a + b
    if q > 2:
        result = a
    if q >= 3:
        result = 0

    return result

@ti.func
def W(r_a: float, r_b: float, h: float) -> float:
    """Kernel smoothing function"""

    r = tm.distance(r_a, r_b)
    q = r/h

    # h**2 since we're in 2D
    return sigma * (1/h**2) * M6(q)

@ti.func
def W_dh(r_a: float, r_b: float, h: float) -> float:
    """Partial derivative of W wrt h"""
    r = tm.distance(r_a, r_b)
    q = r/h

    return (-sigma * (1/h**3)) * (2*M6(q) + q*M6_dq(q))

@ti.func
def W_grad(r_a: float, r_b: float, h: float) -> vec2: # type: ignore
    """Gradient of W"""
    r = tm.distance(r_a, r_b)
    q = r/h

    return r_hat_calc(r_a, r_b) * (sigma / h**3) * M6_dq(q)


@ti.func
def update_dynamic_neighbours(i_a: int, R_kern: float, h_a: float) -> int:
    """
    Updates the dynamic list to conatin the neighbours of the particle at the given index
    
    WARNING: This does NOT clear the dynamic list afterwards, and you must do that manually afterwards
    """
    # Clear the dynamic list
    dynamic_list.deactivate()

    r_a = particle_field[i_a].r

    # Get neighbours
    for i in range(n):
        if i != i_a:
            r_b = particle_field[i].r
            d = tm.distance(r_a, r_b)
            if d < R_kern * h_a:
                dynamic_list.append(particle_field[i].i)
    
    return dynamic_list.length()

@ti.func
def update_dynamic_neighbours_point(r: vec2, R_kern: float, h_a: float) -> int: # type: ignore
    # Get neighbours
    for i in range(n):
        r_b = particle_field[i].r
        d = tm.distance(r, r_b)
        if d < R_kern * h_a:
            dynamic_list2.append(particle_field[i].i)

    return dynamic_list2.length()

"""
Evolution functions
"""

@ti.func
def update_particle_h_and_omega(i_a: int, R_kern: float, h_fact: float):
    # Important values for the given i_a
    r_a = particle_field[i_a].r
    m_a = particle_field[i_a].m
    h_a_0 = particle_field[i_a].h
    
    for _ in range(1): # "Ghost loop" since top level loops are automatically parallelized
        h_a = h_a_0
        O_a = 0.0
        p_sum = 0.0
        error = 1.0
        num_neighbours = 0

        f = 0.0
        f_d = 0.0
        # Newton-Raphson iteration. TODO: Change to a while loop with a condition based on error
        for j in range(10): # This is serialized
            p_sum = 0.0
            W_dh_sum = 0.0

            num_neighbours = update_dynamic_neighbours(i_a, R_kern, h_a)
            for i in range(num_neighbours): # This is serialized
                i_b = dynamic_list[i]
                m_b = particle_field[i_b].m
                r_b = particle_field[i_b].r
                p_sum += m_b * W(r_a, r_b, h_a)
                W_dh_sum += m_b * W_dh(r_a, r_b, h_a)
            dynamic_list.deactivate() # Clear the dynamic list

            p_a = p_calc(m_a, h_a, h_fact)
            O_a = 1 + ((h_a / (2 * p_a)) * W_dh_sum)

            f = p_sum - p_a
            f_d = W_dh_sum + (2 * p_a/h_a)

            h_a_new = h_a - (f/f_d)
            
            error = ti.abs(h_a_new - h_a)/h_a_0

            if (h_a_new < 0):
                print(f'{i_a}: N = {num_neighbours}, error: {error}, h_a: {h_a}, f/f_d: {f/f_d}')
            h_a = h_a_new

            if error < 0.001:
                break
        # print(f'{i_a}: N = {num_neighbours}, error: {error}, h_a: {h_a}, f/f_d: {f/f_d}')
        particle_field[i_a].h = h_a
        particle_field[i_a].O = O_a

@ti.func
def update_particle_v_and_u(i_a: int, R_kern: float, h_fact: float, gamma: float, dt: float):
    # Get needed values from the particle
    r_a = particle_field[i_a].r
    v_a = particle_field[i_a].v
    h_a = particle_field[i_a].h
    m_a = particle_field[i_a].m
    u_a = particle_field[i_a].u
    O_a = particle_field[i_a].O
    p_a = p_calc(m_a, h_a, h_fact)
    P_a = P_calc(p_a, u_a, gamma)

    # Get neighbours
    num_neighbours = update_dynamic_neighbours(i_a, R_kern, h_a)
    
    for _ in range(1): # "Ghost loop" since top level loops are automatically parallelized
        dv_sum = vec2(0.0, 0.0)
        du_sum = 0.0
        for i in range(num_neighbours):
            # Get the particle's index from the dynamic list
            i_b = dynamic_list[i]

            r_b = particle_field[i_b].r
            v_b = particle_field[i_b].v
            h_b = particle_field[i_b].h
            m_b = particle_field[i_b].m
            u_b = particle_field[i_b].u
            O_b = particle_field[i_b].O
            p_b = p_calc(m_b, h_b, h_fact)
            P_b = P_calc(p_b, u_b, gamma)

            # Non-dissipative
            # a = (P_a / (O_a * p_a**2)) * W_grad(r_a, r_b, h_a)
            # b = (P_b / (O_b * p_b**2)) * W_grad(r_a, r_b, h_b)
            # dv_sum += m_b * (a + b)
            # du_sum +=  m_b * tm.dot(v_a - v_b, W_grad(r_a, r_b, h_a))

            # With dissipation
            alpha = 1.0
            alpha_u = 1.0
            beta = 2.0

            r_hat = r_hat_calc(r_a, r_b)
            p_avg = (p_a + p_b)/2
            W_grad_avg = (W_grad(r_a, r_b, h_a) + W_grad(r_a, r_b, h_b))/2
            
            v_sig = v_sig_calc(r_a, r_b, cs_calc(gamma, P_a, p_a), cs_calc(gamma, P_b, p_b), v_a, v_b, beta)
            v_sig_u = v_sig_u_calc(P_a, P_b, p_a, p_b)
            
            v_ab = v_a - v_b
            F_ab = tm.dot(r_hat, W_grad_avg)
            
            dv_sum += m_b * ((alpha * v_sig * v_ab)/p_avg) * F_ab
            du_sum += -(m_b/p_avg) * ((0.5 * alpha * v_sig * tm.dot(v_ab, r_hat)) + (alpha_u * v_sig_u * (u_a - u_b))) * F_ab
        dynamic_list.deactivate()

        dv = dt * (-dv_sum)
        particle_field[i_a].v += dv

        du = dt * ((P_a / (O_a * p_a**2)) * du_sum)
        particle_field[i_a].u += du

        dr = dt * particle_field[i_a].v
        particle_field[i_a].r += dr

        # Boundary conditions
        if particle_field[i_a].r[0] > 1:
            particle_field[i_a].r[0] = 1
            particle_field[i_a].v[0] *= -1
        if particle_field[i_a].r[1] > 1:
            particle_field[i_a].r[1] = 1
            particle_field[i_a].v[1] *= -1
        if particle_field[i_a].r[0] < 0:
            particle_field[i_a].r[0] = 0
            particle_field[i_a].v[0] *= -1
        if particle_field[i_a].r[1] < 0:
            particle_field[i_a].r[1] = 0
            particle_field[i_a].v[1] *= -1

@ti.kernel
def update_all(R_kern: float, h_fact: float, gamma: float, dt: float):
    for _ in range(1):
        for i in range(n):
            update_particle_h_and_omega(i, R_kern, h_fact)
        for i in range(n):
            update_particle_v_and_u(i, R_kern, h_fact, gamma, dt)

"""
Initialization
"""
@ti.kernel
def init(R_kern: float, h_fact: float):
    # Random position for every particle
    # Using ideal gas equation of state (i.e. P = (gamma - 1)*p*u, gamma is the adiabatic index set to 5/3)
    # Using sound speed = 1 (i.e. 1 = (gamma*P/p)**(1/2))
    # This gives u = 0.9
    for i in range(n):
        particle_field[i].i = i
        particle_field[i].r = [ti.random(float), ti.random(float)]
        particle_field[i].v = [0.0, 0.0]
        particle_field[i].h = 0.02
        particle_field[i].m = 1.0
        particle_field[i].u = 0.9
    pass


image = ti.field(dtype=ti.f32, shape=(width, height))
@ti.kernel
def temp(R_kern: float, h_fact: float):
    h = 0.025
    for _ in range(1):
        for x in range(width):
            for y in range(height):
                r = vec2(x/width, y/width)
                num_neighbours = update_dynamic_neighbours_point(r, R_kern, h)

                p_sum = 0.0
                for i in range(num_neighbours):
                    i_b = dynamic_list2[i]
                    r_b = particle_field[i_b].r
                    m_b = particle_field[i_b].m
                    p_sum += m_b * W(r, r_b, h)
                    pass
                dynamic_list2.deactivate()
                image[x, y] = p_sum/4000
    pass
"""
Python scope/Visualization
"""

# All the update calls in one
def update(R_kern: float, h_fact: float, gamma: float, dt: float):
    update_all(R_kern, h_fact, gamma, dt)

# Draws on the canvas
def draw(window: ti.ui.Window, canvas: ti.ui.Canvas, R_kern: float, h_fact: float, dt: float, count: int):
    particle_draw_radius = 0.002
    # if count % 1000 == 0:
    #     temp(R_kern, h_fact)
    # canvas.set_image(image)
    canvas.circles(particle_field.r, particle_draw_radius, (0.1, 0.1, 0.1))
    window.show()

def main(R_kern: float, h_fact: float, gamma: float, dt: float):
    init(R_kern, h_fact)

    window = ti.ui.Window("SPH", res=(width, height), fps_limit=500)
    canvas = window.get_canvas()
    canvas.set_background_color((0.9, 0.9, 0.9))

    count = 0
    while window.running:
        update(R_kern, h_fact, gamma, dt)
        draw(window, canvas, R_kern, h_fact, dt, count)
        count += 1

if __name__ == '__main__':
    # Simulation parameters
    R_kern = 3.0
    h_fact = 1.1
    gamma  = 5/3 # adiabatic index
    dt     = 0.0005
    main(R_kern, h_fact, gamma, dt)