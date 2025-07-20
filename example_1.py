# AirfoiLib - example 1
# This is a detailed tutorial example of meshing the Eppler-387 airfoil with the help of this library.
# It introduces the basics of using the library by creating a ~15k element mesh for the airfoil.
import airfoilib as afl

# ----- GEOMETRY GENERATION -----
# Constants used for the generation of the geometry
file_name = 'e387-il'                  # csv airfoil coordinate file name (Airfoil tools format)
geo_fid = 400                          # geometric fidelity
vertex_i = [0]                         # the indexes of the vertexes of the airfoil, that we want to round off, in this case there's only the trailing edge        
roundlen = 1                           # the length of the depth that round will go at
roundness = 2                          # roundness of the edge, the greater the int, the rounder it will be
aoa = -5*3.14159/180                   # the angle is given in radians, and actually opposite of the wanted aoa as to follow the usual geometric conventions
rotation_center = [50,0]               # the center of the geometric rotation

# Place the csv file in the script directory before running!
airfoil = afl.file.read_csv(file_name)

# Increase the geometric fidelity
airfoil.geofid(geo_fid)

# Round off trailing edge, this is a rather resource intensive process, speed is sacrificed to ensure robustness and generality
airfoil.round_vertex(vertex_i, roundlen, roundness)

# Create a wing section with the airfoil
section = afl.geo.Section([airfoil])

# Add AoA
section.rotate(rotation_center, aoa)

# Inspect the section
afl.pyplot.grid()
section.plot(show=True, color='k')


# ----- MESH CONFIGURATION -----
# Constants used for the configuration of the mesh
# Control volume parameters
xrange = [-500, 1300]                  # the x range of the control volume
yrange = [-500, 500]                   # the y range of the control volume
inlet_arc = True                       # whether to have circle-arc inlet or flat
# Inflation layer parameters
inflation_dist = 30                    # how far the higher mesh density - inflation layer extends
near_field_spacing = 4.5               # the spacing at the boundary of the inflation layer
far_field_spacing = 100                # the spacing at the boundary of the control volume
# Boundary layer's tangent spacing parameters
default_spacing = 5                    # the default tangent spacing, this number might look large but is brought down quite a lot by the following coefficients
smoothness_coef = 0.97                 # smoothness coefficient, takes values between 0 and 1, this reduces unacceptably rapid mesh spacing fluctuation
surface_len_coef = 0.0                 # boundary surface length coefficient, decreases spacing as the length of the surfaces, calculated from the leading edge, increases
proximity_coef = 0                     # proximity coefficient, decreases the spacing in areas where section geometries are close together, useless in our case since we are dealing with a lone airfoil
vertex_sharpness_coef = 20             # vertex sharpness coefficient, decreases the spacing in areas close to sharp vertexes, different coeeficients can be given in a list-pair for convex and reflex angles
curvature_coef = 100                   # curvature coefficient, decreases the spacing in areas of higher curvature, different coeeficients can be given in a list-pair for convex and reflex curvatures
orientation_coef = [6,4]               # orientation coefficient, decreases the spacing in areas oriented away from or towards the inlet, different coeeficients can be given in a list-pair for each case
# Boundary layer's normal spacing parameters
first_cell_height = 0.01               # the first cell height of the boundary layer
thickness = 2                          # the thickness of the boundary layer
growth_ratio = 1.15                    # the growth ratio of the boundary layer
# Wake region parameters
deflection_range = 120                 # deflection distance of the flow from the trailing edge
aoa = [-7*3.14159/180, 7*3.14159/180]  # AoA range that will be investigated, unlike the geometry section this is given with the usual aerodynamic sign. If a range is not needed, a single float can be given instead
far_wake_spacing = 50                  # the spacing at the wake region of the outlet
distribution_degree = 4                # wake spacing distribution degree, higher numbers result in lower spacings, even far away from the trailling edges

# Initialize the domain object
md = afl.Mesh()

# Configure control volume
md.control_volume(xrange, yrange, inlet_arc)

# Configure inflation layer
md.inflation_layer(inflation_dist, near_field_spacing, far_field_spacing)

# Configure boundary layer's tangend spacing
md.tangent_spacing(default_spacing, smoothness_coef, surface_len_coef, proximity_coef, vertex_sharpness_coef, curvature_coef, orientation_coef)

# Configure boundary layer's normal spacing
md.normal_spacing(first_cell_height, thickness, growth_ratio)

# Configure wake region
md.wake_region(deflection_range, aoa, far_wake_spacing, distribution_degree)


# ----- MESH GENERATION -----
# Constants used for the generation of the mesh
model_name = 'eppler-387'              # name of the model and the mesh file (if one is written)
file_extension = '.cgns'               # decides what type of mesh file will be written
review_mesh = True                     # decides if a view of the mesh in gmsh is given
init_gmsh = True                       # decides whether to initialize gmsh or not
fin_gmsh = True                        # decides whether to finalize gmsh or not

# Generate mesh
md.generate(section, model_name, file_extension, review_mesh, init_gmsh, fin_gmsh)
