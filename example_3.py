# This is a further tutorial demonstrating simple geometry manipulation and meshing of a naca 23012 fowler flap
import airfoilib as afl
from math import pi

deflected = True
defang = 30             # deflection angle if flap is deflected, experimental results exist for angles: 0,10,20,30,40

# airfoil
airfoil = afl.file.read_csv('naca23012-il')
airfoil.geofid(300)

# flap
flap = afl.deepcopy(airfoil)


# find flap leading edge location
pf = airfoil.interp(74.36)[1]

# translate to location
flap.scale([0,0], [0.2667, 0.2667])
flap.translate(pf)

# to get the angle of rotation of the flap, we need to find where along the flap, the flap will meet the main airfoil's trailing edge
# so we create an arc and find the intersection of it and the flap
r = afl.geo.distance([pf], [airfoil.te])[0,0]                                  # find  radius of arc
p1 = afl.geo.translate(pf, [r,0])                                              # get second point of arc
arc = afl.geo.circle.arc(airfoil.te, p1, pf, 10**3)
pint, _, flinti = afl.geo.curve.intersect(arc, flap.curves[0], True)[0]        # gather intersection data

rotang = afl.geo.vector.angle(airfoil.te-pf, pint-pf)

flap.rotate(pf,rotang)

if not deflected:
    # get the curve of the flap exposed to air
    fc1 = flap.curves[0][0:flinti+1]
    fc1 = afl.geo.curve.addpoints(fc1, [airfoil.te], [1])

    # get the intersection of the airfoil curve and flap
    pint, flinti, afinti = afl.geo.curve.intersect(flap.curves[0], airfoil.curves[0])[-1]

    # get the curve of the airfoil exposed to air
    ac1 = airfoil.curves[0][0:afinti+1]
    ac1 = afl.geo.curve.addpoints(ac1, [pint], [1])

    # get the second curve of the flap exposed to air
    fc2 = flap.curves[0][flinti+1:]
    fc2 = afl.geo.curve.addpoints(fc2, [pint], [0])

    # assemble the modified airfoil
    airfoil = afl.Airfoil([fc1, ac1, fc2, flap.curves[1]])

    airfoil.geofid(500)
    section = afl.geo.Section([airfoil])
    # Warning: since the reflex edges cannot yet be handled properly by the meshing algorithm of gmsh, this might produce an error in meshing.
    # If this error occures try lowering mesh density and playing around with coefficients.
    # To avoid this error, the boundary layer mesh must be made to not include the area close to the reflex edge, but this is not yet implemented

elif deflected:
    # get the curve of the flap, past the main airfoils trailing edge
    fc = flap.curves[0][flinti+1:]
    
    # get the intersection of the slot curve and the main airfoil curve
    pint, flinti, afinti = afl.geo.curve.intersect(fc, airfoil.curves[0], True)[0]

    # get the airfoil curve exposed to air
    ac = airfoil.curves[0][0:afinti+1]
    ac = afl.geo.curve.addpoints(ac, [pint], [1])

    # get the slot curve
    sc = fc[0:flinti+1]
    sc = afl.geo.curve.addpoints(sc, [airfoil.te, pint], [0,1])

    # assemble modified airfoil
    airfoil = afl.Airfoil([ac, sc])
    
    # translate and rotate flap
    flap.rotate(pf, -rotang)
    hinge =  afl.geo.translate(airfoil.te, [0, -2.5])
    noseloc = afl.geo.translate(hinge, [-1.58*0.2667, 0])
    flap.translate(noseloc-pf)
    flap.rotate(hinge, -defang * pi/180)

    airfoil.geofid(500)
    flap.geofid(500)

    # cut main airfoil trailing edge
    airfoil.round_vertex([1], 0.5, 0)

    section = afl.geo.Section([airfoil, flap])


# scale section
section.scale([0,0], [0.9144, 0.9144])

# rotate around moment calculation point
# section.rotate([23.9*0.9144,-1.6*0.9144], 0*-pi/180)

# inspect section
if True:
    section.plot(True)


md = afl.Mesh()
md.control_volume([-900, 1860], [-900, 900], True)
md.inflation_layer(60, 8.5, 190)
md.tangent_spacing(19, 0.8, 0.2, 0.5, 2, 65, [70,26])
md.normal_spacing(0.000014, 2, 1.16)
md.wake_region(70, [-10* pi / 180, 10* pi / 180], 78, 14)
md.generate(section, 'naca23012_fowler_RENAME', '.cgns', review=True)



