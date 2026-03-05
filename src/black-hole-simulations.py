import taichi as ti # pyright: ignore[reportMissingImports]
import taichi.math as tm # pyright: ignore[reportMissingImports]
ti.init(arch=ti.cuda, default_ip=ti.i32, default_fp=ti.f64)


#########################
# SIMULATION PARAMETERS #
#########################

N: float = 500000
"""Number of particles to simulate around the black hole"""

c: float = 1
"""Speed of light, 1 is simplest"""

G: float = 1
"""Gravitational constant, 1 is simplest"""

M: float = 1
"""Mass of the black hole, 1 is simplest"""

r_s: float = (2 * G * M) / c**2
"""Calculated Schwarzschild radius (event horizon of the black hole)"""

r_isco: float = 3 * r_s
"""Calculated r_isco (circular orbits are unstable within this radius)"""

r_max: float = 500 * r_s
"""Maximum coordinate r before the particle is freed to be reasigned"""


########################
# SIMULATION VARIABLES #
########################

black_hole_position_xy = ti.Vector.field(2, float, shape=())
"""Position of the black hole"""

particles_a_momentum = ti.field(dtype=float, shape=N)
"""Each particle's angular momentum"""

particles_energy = ti.field(dtype=float, shape=N)
"""Each particle's total Energy"""

particles_mass = ti.field(dtype=float, shape=N)
"""Each particle's mass"""

particles_position_rp = ti.Vector.field(2, dtype=float, shape=N)
"""Each particle's position in radial coordinates r, p (p used instead of φ for convenience)"""

particles_celerity = ti.Vector.field(2, dtype=float, shape=N)
"""Each particle's celerity (proper velocity)"""

particles_proper_time_elapsed = ti.field(dtype=float, shape=N)
"""Each particle's elapsed proper time since it was initialized"""

particles_is_free = ti.field(dtype=bool, shape=N)
"""If each particle is free to be reasigned"""

particles_is_grid = ti.field(dtype=bool, shape=N)
"""If each particle is part of the grid"""


####################
# SCREEN VARIABLES #
####################

screen_black_hole_position = ti.Vector.field(2, float, 1)
"""Position of the black hole relative to the bottom left of the screen"""

screen_particles_position = ti.Vector.field(2, dtype=float, shape=N)
"""Each particle's position relative to the bottom left of the screen"""

screen_particles_color = ti.Vector.field(3, dtype=float, shape=N)
"""Each particle's color RGB value"""

screen_particles_radii = ti.field(dtype=float, shape=N)
"""Each particle's radius to be drawn"""


##########
# UNUSED #
##########
@ti.func
def add_proper_velocites(u: float, v: float) -> float:
    """Add proper velocites (celerity) and return the result"""
    return u + v + ( (gamma(u) / (1 + gamma(u))) * (u * v) + (((1 - gamma(v))/gamma(v)) * u) )

@ti.func
def mu(m: float) -> float:
    """Returns the reduced mass of m relative to M"""
    return ((M * m) / (M + m))


########################
# CALULATION FUNCTIONS #
########################

@ti.func # Done
def gamma(v: float) -> float:
    """Returns the Lorentz Factor of a given v"""
    return 1 / ti.sqrt(1 - (v**2 / c**2))

@ti.func # Done
def rho(r: float) -> float:
    """Returns 1 - (rs / r)"""
    return (1 - (r_s / r))

@ti.func # Done
def compute_d2r(r: float, l: float, m: float) -> float:
    """Returns the proper acceleration of coordinate r, given r, l, and m"""
    L_m = l / m
    if (L_m < 0):
        L_m *= -1
    return (L_m**2 / r**3) - ((G * M / r**2) * (1 + (3 * L_m**2 / (c**2 * r**2))))

@ti.func # Done
def compute_d2p(r: float, dr: float, dp: float) -> float:
    """Returns the proper acceleration of coordinate p, given r, dr, and dp"""
    return -(2 * dr * dp) / r

@ti.func # Done
def compute_ciruclar_orbit_velocity(r: float) -> float:
    """Returns the local velocity required to have a circular orbit at input distance r"""
    return ti.sqrt((G * M) / (r - r_s))

"""
######################
# PARTICLE FUNCTIONS #
######################
"""
@ti.func # Done
def particle_free(i: int):
    """Free a particle to be initialized by reseting all properties of the particle"""
    particles_a_momentum[i] = 0.0
    particles_energy[i] = 0.0
    particles_mass[i] = 0.0
    particles_position_rp[i] = [0.0, 0.0]
    particles_celerity[i] = [0.0, 0.0]
    particles_proper_time_elapsed[i] = 0.0
    particles_is_free[i] = True
    particles_is_grid[i] = False

@ti.func # Done
def particle_print(i: int):
    """Prints all information about a particle"""
    print(f'------- Particle {i} -------')
    print(f'Angular momentum:  {particles_a_momentum[i]}')
    print(f'Total energy:      {particles_energy[i]}')
    print(f'Mass:              {particles_mass[i]}')
    print(f'Coordinates (r,p): ({particles_position_rp[i][0]}, {particles_position_rp[i][1]})')
    print(f'Proper velocity:   {particles_celerity[i]}')
    print(f'Proper time:       {particles_proper_time_elapsed[i]}')
    print(f'is_free:           {particles_is_free[i]}')
    print(f'is_grid:           {particles_is_grid[i]}')
    pass

@ti.func # Done
def particle_init_rp(i: int, m: float, r: float, p: float, v: float, a: float):
    """
    Initializes a particle, given the following parameters:
    
    :param i: Index of the particle
    :type i: int
    :param m: Mass of the particle
    :type m: float
    :param r: Distance coordinate of the particle
    :type r: float
    :param p: Angle coordinate of the particle
    :type p: float
    :param v: Local velocity
    :type v: float
    :param a: Angle of velocity (e.g. 0: tangent to the black hole, pi: straight radially out, -pi: straight radial in)
    :type a: float
    """

    if not particles_is_free[i]:
        print(f'WARN: particle_init_rp(): Particle {i} is not free')

    # Calculate inital values based on inputs
    gamma_ = gamma(v)
    rho_   = rho(r)
    v_r    = v * tm.sin(a) # Radial velocity
    v_t    = v * tm.cos(a) # Transverse velocity
    dr     = v_r * tm.sqrt(rho_) * gamma_
    dp     = v_t * (1 / r) * gamma_

    # Calculate and set properties based on inputs
    particles_a_momentum[i]  = m * v_t * r * gamma_
    particles_energy[i]      = m * c**2 * tm.sqrt(rho_) * gamma_
    particles_mass[i]        = m
    particles_position_rp[i] = [r, p]
    particles_celerity[i]    = [dr, dp]
    particles_is_free[i]     = False
    pass

@ti.func # Done
def particle_init_rp_stationary(i: int, m: float, r: float, p: float):
    """
    Initializes a particle with no velocity, given the following parameters:
    
    :param i: Index of the particle
    :type i: int
    :param m: Mass of the particle
    :type m: float
    :param r: Distance coordinate of the particle
    :type r: float
    :param p: Angle coordinate of the particle
    :type p: float
    """
    particle_init_rp(i, m, r, p, 0, 0)
    pass

@ti.func # Done
def particle_init_rp_circular_orbit(i: int, m: float, r: float, p: float):
    """
    Initializes a particle to a stable circular orbit, given the following parameters:

    :param i: Index of the particle
    :type i: int
    :param m: Mass of the particle
    :type m: float
    :param r: Distance coordinate of the particle
    :type r: float
    :param p: Angle coordinate of the particle
    :type p: float
    """
    # Note: Stable circular orbits do not exist for r < 3*rs, also known as r-isco
    # Note: The circular orbit velocity at r = 3*rs/2 is c
    v = compute_ciruclar_orbit_velocity(r)
    particle_init_rp(i, m, r, p, v, 0)
    pass

@ti.func # Done
def particle_init_rp_random_orbit(i: int, m: float, r1: float, r2: float, p1: float, p2: float, v1: float, v2: float, a1: float, a2: float):
    """
    Initializes a particle to a randomized orbit, given the following parameters:
    
    :param i: Index of the particle
    :type i: int
    :param m: Mass of the particle
    :type m: float
    :param r1: Minimum distance coordinate to randomly select from
    :type r1: float
    :param r2: Maximum distance coordinate to randomly select from
    :type r2: float
    :param p1: Minimum angle coordinate to randomly select from
    :type p1: float
    :param p2: Maximum angle coordinate to randomly select from
    :type p2: float
    :param v1: Minimum local velocity, relative to a circular orbit, to randomly select from (e.g. 0.5: half of circular orbit velocity)
    :type v1: float
    :param v2: Maximum local velocity, relative to a circular orbit, to randomly select from (e.g. 2: twice circular orbit velocity)
    :type v2: float
    :param a1: Minimum angle of velocity to randomly select from (e.g. 0: tangent to the black hole, pi: straight radially out, -pi: straight radial in)
    :type a1: float
    :param a2: Maximum angle of velocity to randomly select from (e.g. 0: tangent to the black hole, pi: straight radially out, -pi: straight radial in)
    :type a2: float
    """
    r = r1 + ((r2 - r1) * ti.random(dtype=float))
    p = p1 + ((p2 - p1) * ti.random(dtype=float))
    v1_ = v1 * compute_ciruclar_orbit_velocity(r)
    v2_ = v2 * compute_ciruclar_orbit_velocity(r)
    v = v1_ + ((v2_ - v1_) * ti.random(dtype=float))
    a = a1 + ((a2 - a1) * ti.random(dtype=float))
    particle_init_rp(i, m, r, p, v, a)
    pass

@ti.func # Done
def particle_init_xy(i: int, m: float, x: float, y: float, v: float, a: float):
    """
    Initializes a particle, given the following parameters:
    
    :param i: Index of the particle
    :type i: int
    :param m: Mass of the particle
    :type m: float
    :param x: x coordinate of the particle, positive x is to the right of the black hole
    :type x: float
    :param y: y coordinate of the particle, positive y is above the black hole
    :type y: float
    :param v: Local velocity
    :type v: float
    :param a: Angle of velocity (0: tangent, pi: straight radially out, -pi: straight radial in)
    :type a: float
    """
    r = tm.sqrt(x**2 + y**2)
    p = tm.atan2(y, x)
    particle_init_rp(i, m, r, p, v, a)
    pass

@ti.func # Done
def particle_init_xy_stationary(i: int, m: float, x: float, y: float):
    """
    Initializes a particle to a point with no velocity, given the following parameters:
    
    :param i: Index of the particle
    :type i: int
    :param m: Mass of the particle
    :type m: float
    :param x: x coordinate of the particle, positive x is to the right of the black hole
    :type x: float
    :param y: y coordinate of the particle, positive y is above the black hole
    :type y: float
    """
    particle_init_xy(i, m, x, y, 0, 0)
    pass


"""
###################
# DISPLAY KERNELS #
###################
"""
# This is the final computation
@ti.kernel # Done
def update_all_dw_rk4(dt: float):
    """Compute acceleration and update the positions of each particle given dt"""
    # Schwarzschild Metric with θ=pi/2
    for i in range(N):
        if (particles_is_free[i]):
            continue

        # Get current radial coordinate
        r = particles_position_rp[i][0]

        if (r <= r_s or r > r_max):
            # Schwarzschild metric with Schwarzschild coordinates is invalid for r <= rs
            particle_free(i)
            continue

        # Get current velocity, angular momentum, and mass
        dr = particles_celerity[i][0]
        dp = particles_celerity[i][1]
        l = particles_a_momentum[i]
        m = particles_mass[i]

        # Calculate current acceleration
        d2r = compute_d2r(r, l, m)
        d2p = compute_d2p(r, dr, dp)

        # Calculate acceleration at next time step using an approximation
        # RK4, aka the Runge-Kutta method
        ### k1
        dr_k1 = dr
        dp_k1 = dp
        d2r_k1 = d2r
        d2p_k1 = d2p
        r_k1 = r
        
        ### k2
        dr_k2 = dr_k1 + (d2r_k1*dt/2)
        dp_k2 = dp_k1 + (d2p_k1*dt/2)
        r_k2 = r_k1  + (dr_k2*(dt/2))
        d2r_k2 = compute_d2r(r_k2, l, m)
        d2p_k2 = compute_d2p(r_k2, dr_k2, dp_k2)
        
        ### k3
        dr_k3 = dr_k2 + (d2r_k2*dt/2)
        dp_k3 = dp_k2 + (d2p_k2*dt/2)
        r_k3 = r_k2  + (dr_k3*dt/2)
        d2r_k3 = compute_d2r(r_k3, l, m)
        d2p_k3 = compute_d2p(r_k3, dr_k3, dp_k3)

        ### k4
        dr_k4 = dr_k3 + (d2r_k3*dt)
        dp_k4 = dp_k3 + (d2p_k3*dt)
        r_k4 = r_k3  + (dr_k4*dt)
        d2r_k4 = compute_d2r(r_k4, l, m)
        d2p_k4 = compute_d2p(r_k4, dr_k4, dp_k4)
        
        ### Final result
        d2r_rk4 = d2r + ((dt/6) * (d2r_k1 + (2*d2r_k2) + (2*d2r_k3) + d2r_k4))
        d2p_rk4 = d2p + ((dt/6) * (d2p_k1 + (2*d2p_k2) + (2*d2p_k3) + d2p_k4))

        ### Apply acceleration
        particles_celerity[i][0] += (d2r_rk4 * dt)
        particles_celerity[i][1] += (d2p_rk4 * dt)

        # Update coordinates
        particles_position_rp[i][0] += (particles_celerity[i][0] * dt)
        particles_position_rp[i][1] += (particles_celerity[i][1] * dt)

        # Keep track of elapsed time
        particles_proper_time_elapsed[i] += dt
    pass

@ti.kernel # Done
def update_screen_positions(camera_x: float, camera_y: float, zoom_factor: float):
    """
    Update the screen position of the particles and black hole, given the following parameters:
    
    :param camera_x: x coordinate of the camera
    :type camera_x: float
    :param camera_y: y coordinate of the camera
    :type camera_y: float
    :param zoom_factor: How zoomed out the camera is, higher = further
    :type zoom_factor: float
    """
    # Convert radial coordinates relative to the black hole to cartesian coordinates
    # Then translate by the camera positions, scale it by zoom_factor,
    # finally translate by 0.5 to set them current position to the centre of the screen
    for i in range(N):
        screen_particles_position[i][0] = ((particles_position_rp[i][0] + black_hole_position_xy[None][0]) * tm.cos(particles_position_rp[i][1]) + camera_x)/zoom_factor + 0.5
        screen_particles_position[i][1] = ((particles_position_rp[i][0] + black_hole_position_xy[None][1]) * tm.sin(particles_position_rp[i][1]) + camera_y)/zoom_factor + 0.5
    
    screen_black_hole_position[0][0] = (black_hole_position_xy[None][0] + camera_x)/zoom_factor + 0.5
    screen_black_hole_position[0][1] = (black_hole_position_xy[None][1] + camera_y)/zoom_factor + 0.5
    pass

@ti.kernel # Done
def update_colors(dt: float):
    """Update colors of each particle given dt"""
    for i in range(N):
        if particles_is_free[i]:
            continue

        # Grid particles
        if particles_is_grid[i]:
            # TODO: Probably better to have a time based decay, but the current effect is good
            if screen_particles_color[i][0] <= 0.2 or screen_particles_color[i][1] <= 0.2 or screen_particles_color[i][2] <= 0.2:
                particle_free(i)
                screen_particles_color[i] = screen_particle_default_color
                continue
            else:
                screen_particles_color[i][0] -= dt * 0.025
                screen_particles_color[i][1] -= dt * 0.025
                screen_particles_color[i][2] -= dt * 0.025
        else: # Non-grid particles
            ### Speed based
            # This is all arbitrary, I just picked what I thought looked good
            dr = particles_celerity[i][0]
            dp = particles_celerity[i][1]
            v = tm.sqrt(dr**2 + dp**2)
            red = (2*v) + 0.1
            red2 = (1*v)**2
            gb = v**1.5 - 0.2
            screen_particles_color[i] = [red + red2, gb, gb]
    pass

@ti.kernel # Done
def set_screen_particles_radii(start_index: int, n: int, r: float):
    """
    Sets a number of particles to the radius, give the following paramaters:
    
    :param start_index: Index of the first particle radius to change
    :type start_index: int
    :param n: Number of particles
    :type n: int
    :param r: Radius
    :type r: float
    """
    for i in range(n):
        screen_particles_radii[i + start_index] = r


@ti.kernel # Done
def particle_circle(start_index: int, n: int, x: float, y: float, r: float, v: float, a: float) -> int:
    """
    Spawn a particle circle, given the following parameters:
    
    :param start_index: Index of the first particle to use
    :type start_index: int
    :param n: Number of particles
    :type n: int
    :param x: x coordinate for the centre of the circle
    :type x: float
    :param y: y coordinate for the centre of the circle
    :type y: float
    :param r: Radius of the circle
    :type r: float
    :param v: Local velocity to give each particle
    :type v: float
    :param a: Angle of local velocity to give each particle
    :type a: float
    :return: Index of the first particle after those used in this circle
    :rtype: int
    """
    for i in range(n):
        # Random polar coordinates relative to the x_, y_
        r_ = ti.random(dtype=float) * r
        p_ = ti.random(dtype=float) * tm.pi * 2
        x_i = x + (r_ * tm.cos(p_))
        y_i = y + (r_ * tm.sin(p_))
        particle_init_xy(i + start_index, 1, x_i, y_i, v, a)
    return start_index + n


@ti.func # Done
def particles_grid_line_x(start_index: int, n: int, x: float, s: float, variance: float):
    """
    Initializes a vertical line of particles, given the following parameters:
    
    :param start_index: Index of the first particle to initialize
    :type start_index: int
    :param n: Number of particles to initialize
    :type n: int
    :param x: x coordinate of the line, positive x is to the right of the black hole
    :type x: float
    :param s: Space between each particle
    :type s: float
    :param variance: How varied each particles position will be relative to the idealized line
    :type variance: float
    """
    for i in range(n):
        if not particles_is_grid[i + start_index]:
            particles_is_grid[i + start_index] = True
        y = (i-(n/2)) * s * (ti.random(dtype=float) * variance + (1 - (variance/2)))
        particle_init_xy_stationary(i + start_index, 1, x, y)
        screen_particles_color[i + start_index] = screen_particle_default_color
    pass

@ti.func # Done
def particles_grid_line_y(start_index: int, n: int, y: float, s: float, variance: float):
    """
    Initializes a vertical line of particles, given the following parameters:
    
    :param start_index: Index of the first particle to initialize
    :type start_index: int
    :param n: Number of particles to initialize
    :type n: int
    :param y: x coordinate of the line, positive x is to the right of the black hole
    :type y: float
    :param s: Space between each particle
    :type s: float
    :param variance: How varied each particle's position will be relative to the idealized line
    :type variance: float
    """
    for i in range(n):
        if not particles_is_grid[i + start_index]:
            particles_is_grid[i + start_index] = True
        x = (i-(n/2)) * s * (ti.random(dtype=float) * variance + (1 - (variance/2)))
        particle_init_xy_stationary(i + start_index, 1, x, y)
        screen_particles_color[i + start_index] = screen_particle_default_color
    pass

@ti.kernel # Done
def particles_grid(start_index: int, num_lines: int, n_per_line: int, line_spacing: float, s: float, variance: float) -> int:
    """
    Creates a grid of particles, given the following paramaters
    
    :param start_index: Index of the first particle to initialize
    :type start_index: int
    :param num_lines: Number of grid lines to initialize
    :type num_lines: int
    :param n_per_line: Number of particles per grid line
    :type n_per_line: int
    :param line_spacing: Space between each grid line
    :type line_spacing: float
    :param s: Space between each particle in each grid line
    :type s: float
    :param variance: How varied each particle's position will be relative to the idealized line
    :type variance: float
    :return: Index of the first particle after those used in this grid
    :rtype: int
    """
    for x in range(num_lines):
        particles_grid_line_x(start_index + (x * n_per_line), n_per_line, (x-(num_lines/2)) * line_spacing + (line_spacing/2), s, variance)

    y_start_index = start_index + (num_lines * n_per_line)
    for y in range(num_lines):
        particles_grid_line_y(y_start_index + (y * n_per_line), n_per_line, (y-(num_lines/2)) * line_spacing + (line_spacing/2), s, variance)
    
    return start_index + (2 * num_lines * n_per_line)

"""
DISPLAY PARAMETERS, in order:
- Size of the window
- Radius of each particle
- Background color
- Whether to show r-isco or not
- Whether to show max_r or not
"""
screen_width, screen_height    = 900, 900
screen_particle_default_color  = (0.7, 0.7, 0.7)
screen_particle_default_radius = 0.00075
screen_background_color        = (0.2,0.2,0.2)
screen_show_r_isco = False
screen_show_max_r  = False

@ti.kernel # Done
def init() -> int:
    """Initialize the simulation, then return the total number of particles available"""
    black_hole_position_xy[None] = (0, 0)
    for i in range(N):
        particle_free(i)

    screen_black_hole_position[0] = (0,0)
    screen_particles_position.fill((0,0))
    screen_particles_color.fill(screen_particle_default_color)
    screen_particles_radii.fill(screen_particle_default_radius)

    return N

"""
#################
# MAIN FUNCTION #
#################
"""
def main():
    # Initialize the Taichi window and canvas
    window = ti.ui.Window("General Relativity", res=(screen_width, screen_height), fps_limit=500)
    canvas = window.get_canvas()
    canvas.set_background_color(screen_background_color)

    # Control variables
    camera_pos  = [0.5, 0.5]
    zoom_factor = 80
    dt          = 1
    substeps    = 10

    # Control parameters
    camera_control_size = 0.5
    zoom_control_size   = 0.5
    # dt_control_size     = 0.005 # NOTE: Only use for debugging as changing dt during simulation can cause inaccuracy
    zoom_min = 1
    zoom_max = 100
    # dt_min   = 0.005 # NOTE: Only use for debugging as changing dt during simulation can cause inaccuracy
    # dt_max   = 1.5 # NOTE: Only use for debugging as changing dt during simulation can cause inaccuracy


    # Initialize the simulation and perpare particle count tracking
    total_particles = init()
    used_particles  = 0

    ### Spawn some particles in!
    
    ### Respawning Grid
    # Grid parameters
    grid_num_lines        = 20 # Looks better when this is even
    grid_n_per_line       = 400
    grid_line_spacing     = 2 * r_s
    grid_particle_spacing = grid_line_spacing * grid_num_lines / grid_n_per_line
    grid_line_variance    = 0.002
    
    # These have to be manually tweaked to not reasign too early
    grid_respawn_count       = 10
    grid_respawn_interval    = 2
    grid_particles_per_spawn = grid_num_lines * grid_n_per_line * 2

    # Keep track of the indices for the grid
    grid_start_index   = 0 # Must start at zero for the modulus math to work
    current_grid_index = particles_grid(grid_start_index, grid_num_lines, grid_n_per_line, grid_line_spacing, grid_particle_spacing, grid_line_variance)

    # Calculate the index of the first non-grid particle
    next_index         = grid_particles_per_spawn * (grid_respawn_count + 1) # Add one to count the initial spawn
    used_particles    += next_index


    ### Large particle circle
    circle_start_index = next_index
    circle_n           = 100000
    next_index         = particle_circle(circle_start_index, circle_n, 15, 15, 2, 0.21825, -tm.pi/6)
    used_particles    += circle_n
    set_screen_particles_radii(circle_start_index, circle_n, 0.00175)


    # Print information about the current run
    print()
    print('----------- Black Hole Simulation -----------')
    print(f'Total particles: {N}')
    print(f'Particles assigned: {used_particles}')
    print(f'dt: {dt} | Substeps: {substeps}')
    print()


    # Time tracking variables, mostly for grid respawning
    elapsed_time = 0
    grid_last_spawn_time = 0

    # Main loop
    while window.running:
        # Controls
        if window.is_pressed('w'): camera_pos[1] -= camera_control_size
        if window.is_pressed('a'): camera_pos[0] += camera_control_size
        if window.is_pressed('s'): camera_pos[1] += camera_control_size
        if window.is_pressed('d'): camera_pos[0] -= camera_control_size
        if window.is_pressed('e'): zoom_factor -= zoom_control_size
        if window.is_pressed('q'): zoom_factor += zoom_control_size
        # if window.is_pressed('z'): dt -= dt_control_size # NOTE: Only use for debugging as changing dt during simulation can cause inaccuracy
        # if window.is_pressed('c'): dt += dt_control_size # NOTE: Only use for debugging as changing dt during simulation can cause inaccuracy

        if zoom_factor <= zoom_min: zoom_factor = zoom_min
        if zoom_factor >= zoom_max: zoom_factor = zoom_max
        # if dt <= dt_min: dt = dt_min # NOTE: Only use for debugging as changing dt during simulation can cause inaccuracy
        # if dt >= dt_max: dt = dt_max # NOTE: Only use for debugging as changing dt during simulation can cause inaccuracy


        # Main simulation steps
        for _ in range(substeps):
            update_all_dw_rk4(dt/substeps)
        update_colors(dt)
        update_screen_positions(camera_pos[0], camera_pos[1], zoom_factor)

        # Optional display parameters
        if screen_show_max_r:
            canvas.circles(black_hole_position_xy, r_max/zoom_factor, (0.3,0.3,0.3))
        if screen_show_r_isco:
            canvas.circles(black_hole_position_xy, r_isco/zoom_factor, (0.5,0.5,0.5))

        # Draw the particles and black hole
        canvas.circles(screen_particles_position, screen_particle_default_radius, per_vertex_color=screen_particles_color, per_vertex_radius=screen_particles_radii)
        canvas.circles(screen_black_hole_position, r_s/zoom_factor, (0,0,0))

        # Keep track of time
        elapsed_time += dt

        # Respawn the grid periodically
        if elapsed_time - grid_last_spawn_time >= grid_respawn_interval:
            current_grid_index = particles_grid(current_grid_index, grid_num_lines, grid_n_per_line, grid_line_spacing, grid_particle_spacing, grid_line_variance)
            grid_last_spawn_time = elapsed_time
            current_grid_index %= (grid_particles_per_spawn * (grid_respawn_count + 1)) # Add one to count the initial spawn
        
        # Finally
        window.show()
    pass

if __name__ == "__main__":
    main()
    pass