
import airfoilib as afl
from math import pi
from time import time

# Basic Configuration
measure_time = False
name = 'naca23012-il'

# Optioning
showdiag = not measure_time
t1 = time()

# Read ordinates
airfoil = afl.file.read_csv(name)

# Create weight pairs for shortcut function
w = afl.geo.spline.bezier([[0.1, 0], [0.3, 0.1], [0.35, 0.7], [0.25, 1]])
w.delta = 1/30
w = w.evalpts

# Create main flap
sc = afl.geo.crvforge.shortcut(airfoil.profile, w)
airfoil.geofid(300)
mainfoil, flap = airfoil.slot(90,75,input_curve=sc)
flap.geofid(300)

# Create auxilary flap
mainflap, auxiflap = flap.slot(94, 90, forge_funct=afl.geo.crvforge.arcross, forge_args=[])

# Create slat
slat, mainfoil = mainfoil.slot(15, 3, forge_funct=afl.geo.crvforge.arcross, forge_args=[])

# Simplify
auxiflap.fillet([1], 'arc', 0.5)
mainfoil.geofid(300)
mainflap.geofid(300)
auxiflap.geofid(300)
slat.geofid(300)
mainfoil.round_vertex([2,3], 0.5, 0)
mainflap.round_vertex([3], 0.5, 0)
slat.round_vertex([1], 0.5, 0)

# Move into extended position
slat.translate([-12, -10])
slat.rotate(slat.le, 10*pi/180)
mainflap.translate([19, -1])
mainflap.rotate(mainflap.le, -30*pi/180)
auxiflap.translate([19, -1])
auxiflap.rotate(mainflap.le, -30*pi/180)
auxiflap.translate([3, -1])
auxiflap.rotate(auxiflap.le, -20*pi/180)

# Create Section
section = afl.geo.Section([slat, mainfoil, mainflap, auxiflap])
t2= time()

# Preview
section.plot(showdiag, color='k')

# Configure Mesh
md = afl.Mesh()
md.control_volume([-900, 1860], [-900, 900], True)
md.inflation_layer(60, 7, 190)
md.tangent_spacing(15, 0.97, 0.12, 1.1, 20, 70, [20,40])
md.normal_spacing(0.0005, 1.4, 1.16)
md.wake_region(200, [-4*pi/180, 4*pi/180], 80, 14)

# Generate mesh
md.generate(section, 'naca23012_funky', '', review=showdiag)

if measure_time:
    print(t2-t1)
    print(time()-t1)
    print(time()-t2)
