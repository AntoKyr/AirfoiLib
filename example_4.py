
import airfoilib as afl
from math import pi
from time import time

t1 = time()
airfoil = afl.file.read_csv('naca23012-il')

w = afl.geo.spline.bezier([[0.1, 0], [0.3, 0.1], [0.35, 0.7], [0.25, 1]])
w.delta = 1/30
w = w.evalpts
sc = afl.geo.crvforge.shortcut(airfoil.profile, w)
airfoil.geofid(300)
mainfoil, flap = airfoil.slot(90,75,input_curve=sc)
flap.geofid(300)
mainflap, auxiflap = flap.slot(94, 90, forge_funct=afl.geo.crvforge.arcross, forge_args=[])
slat, mainfoil = mainfoil.slot(15, 3, forge_funct=afl.geo.crvforge.arcross, forge_args=[[26,6]])

auxiflap.fillet([1], 'arc', 0.5)
mainfoil.geofid(300)
mainflap.geofid(300)
auxiflap.geofid(300)
slat.geofid(300)
mainfoil.round_vertex([2,3], 0.5, 0)
mainflap.round_vertex([3], 0.5, 0)
slat.round_vertex([1], 0.5, 0)

slat.translate([-12, -10])
slat.rotate(slat.le, 10*pi/180)
mainflap.translate([19, 0])
mainflap.rotate(mainflap.le, -30*pi/180)
auxiflap.translate([19, 0])
auxiflap.rotate(mainflap.le, -30*pi/180)
auxiflap.translate([3, -1])
auxiflap.rotate(auxiflap.le, -20*pi/180)

section = afl.geo.Section([slat, mainfoil, mainflap, auxiflap])
t2= time()
print(t2-t1)

md = afl.Mesh()
md.control_volume([-900, 1860], [-900, 900], True)
md.inflation_layer(60, 7, 190)
md.tangent_spacing(7, 0.85, 0.2, 0.5, 5, 55, [50,35])
md.normal_spacing(0.0001, 1.6, 1.16)
md.wake_region(200, [-4*pi/180, 4*pi/180], 80, 14)
md.generate(section, 'naca23012_funky', '', review=False)

print(time()-t1)
print(time()-t2)
