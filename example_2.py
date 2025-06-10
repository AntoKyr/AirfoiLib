# This is a tutorial demonstrating basics of section geometry generation and manipulation.

import airfoilib as afl
import numpy as np
from time import time

# GEOMETRY GENERATION
main_flap = True            # if True, create a slotted main flap
aux_flap = False             # if True and main_flap True, create a slotted auxilary flap
mfpi = 3                    # path index of main flap (0-6 for only main_flap, 0-2 for both main and auxilary flap)
afpi = 3                    # path index of auxilary flap (for mfpi 0-1: 0-5, for mfpi 2: 0-4)

# airfoil
airfoil = afl.file.read_csv('naca23012-il')
airfoil.geofid(300)         # its always a good idea to increase geometric fidelity before starting to manipulate a geometry

if not main_flap:
    section = afl.geo.Section([airfoil])

else:
    # main flap coordinatess
    ss = [[0,-1.29], [0.4,-0.32], [0.72,0.04], [1.36,0.61], [2,1.04], [2.64,1.4], [3.92,1.94], [5.2,2.3], [6.48,2.53], [7.76,2.63], [9.03,2.58], [10.31,2.46], [15.66,1.68], [20.66,0.92], [25.66, 0.13]]
    ps = [[0,-1.29], [0.4,-2.05], [0.72,-2.21], [1.36,-2.36], [2,-2.41], [2.64,-2.41], [5.66,-2.16], [15.66,-1.23], [20.66,-0.7], [25.66, -0.13]]
    # arrange and patch the curves of the main flap
    fc = afl.geo.curve.patch(afl.geo.curve.arrange(ss,ps, 10**-3))
    # translate main flap to correct position
    fc = afl.geo.translate(fc, [100-25.66,0])
    # make an airfoil out of the main flap
    main_flap = afl.Airfoil([fc])

    # main slot coordinates
    sc = [[72.32,-1.02], [74.57,0.67], [76.32,1.76], [77.82,2.3], [79.32,2.65], [80.82,2.82], [82.7, 2.64]]

    # morphology of slot
    p0 = [66.65, 4.67]                                                           # center of arc of slot
    i1, pp1 = afl.geo.curve.proxi2p(airfoil.curves[0], sc[-1])                   # find where to cut main airfoil on the suction side, so it fits the slot
    i2, ptan2, r2 = afl.geo.circle.tang2crv(airfoil.curves[0], p0, 7.97)         # find where to cut main airfoil on the pressure side, so it fits the slot
    airfoil_c = airfoil.curves[0][i1+1:i2]
    airfoil_c = afl.geo.curve.addpoints(airfoil_c, [pp1, ptan2], [0,1])          # add points to airfoil main curve
    arc = afl.geo.circle.arc(ptan2, sc[0], p0, 5)                                # create arc between slot and airfoil

    # patch the curves together
    airfoil_c = afl.geo.curve.patch([airfoil_c, arc, sc])

    # generate section
    airfoil = afl.Airfoil([airfoil_c])
    airfoil.geofid(300)
    main_flap.geofid(300)
    
    if not aux_flap:
        # flap deflection
        # path of main flap
        mfle = [74.34, -1.29]
        afte = [82.7, 2.64]
        path = np.array([[0,8.36,3.91], [10,5.41,3.63], [20,3.83,3.45], [30,2.63,3.37], [40,1.35,2.43], [50,0.5,1.63], [60, 0.12, 1.48]])
        path[:,1:3] = np.tile(afte, (len(path),1)) - path[:,1:3]
        # translate and rotate flap
        main_flap.rotate(mfle, -path[mfpi,0]*np.pi/180)
        main_flap.translate(path[mfpi,1:3]-mfle)
        section = afl.geo.Section([airfoil, main_flap])

    else:
        # auxilary flap
        ss = [[0,-0.43], [0.25,0.06], [0.5,0.31], [0.75,0.49], [1,0.63], [1.5,0.85], [2,0.99], [2.5,1.08], [3,1.12], [3.5,1.12], [4,1.06], [5,0.92], [10, 0.13]]
        ps = [[0,-0.43], [0.25,-0.84], [0.5,-0.92], [0.75,-0.98], [1,-1], [1.5,-1.02], [2,-1], [2.5,-0.96], [5,-0.7], [10, -0.13]]
        # arrange and patch the curves of the auxilary flap
        fc = afl.geo.curve.patch(afl.geo.curve.arrange(ss,ps, 10**-3))
        fc = afl.geo.translate(fc, [100-10,0])
        # make an airfoil out of the auxilary flap
        aux_flap = afl.Airfoil([fc])

        # auxilary slot
        sc = [[88.5,-0.86], [89,-0.48], [90, 0.28], [90.5,0.66], [91,0.9], [91.5,1.05], [92,1.17], [92.5,1.22], [93,1.15], [93.23, 1.13]]

        # morphology of slot
        p0 = [86.59,1.52]
        i1, pp1 = afl.geo.curve.proxi2p(main_flap.curves[0], sc[-1])
        i2, ptan2, r2 = afl.geo.circle.tang2crv(main_flap.curves[0], p0, 3.05)

        flap_c = main_flap.curves[0][i1+1:i2]
        flap_c = afl.geo.curve.addpoints(flap_c, [pp1, ptan2], [0,1])
        arc = afl.geo.circle.arc(ptan2, sc[0], p0, 5)

        flap_c = afl.geo.curve.patch([flap_c, arc, sc])

        # generate section
        main_flap = afl.Airfoil([flap_c])
        main_flap.geofid(300)
        aux_flap.geofid(300)

        # flap deflection-----------
        # path of main flap
        mfle = [74.34, -1.29]
        afte = [82.7, 2.64]
        m_path = np.array([[0,8.36,3.91], [20,3.83,3.45], [40,1.35,2.43]])
        m_path[:,1:3] = np.tile(afte, (len(m_path),1)) - m_path[:,1:3]
        
        # path of auxilary flap
        afle = [90, -0.43]
        mfte = main_flap.curves[0][-1]
        if mfpi == 0:
            a_path = np.array([[0,3.22,1.58], [10,0.55,2.77], [20,0.32,2.5], [30,0.06,1.27], [40,0.25,0.59], [50,0.42,0.47]])
            a_path[:,1:3] = np.tile(mfte, (len(a_path),1)) - a_path[:,1:3]
        if mfpi == 1:
            a_path = np.array([[0,3.22,1.58], [10,1.55,1.52], [20,1.32,1.5], [30,1.06,1.5], [40,0.75,0.59], [50,0.42,0.47]])
            a_path[:,1:3] = np.tile(mfte, (len(a_path),1)) - a_path[:,1:3]
        if mfpi == 2:
            a_path = np.array([[0,3.22,1.58], [10,0.55,1.77], [20,0.32,1.5], [30,0.06,1.27], [40,0.25,0.59]])
            a_path[:,1:3] = np.tile(mfte, (len(a_path),1)) - a_path[:,1:3]

        # translate and rotate auxilary flap
        aux_flap.rotate(afle, -a_path[afpi,0]*np.pi/180)
        aux_flap.translate(a_path[afpi,1:3]-afle)

        # translate and rotate flap and auxilary
        main_flap.rotate(mfle, -m_path[mfpi,0]*np.pi/180)
        main_flap.translate(m_path[mfpi,1:3]-mfle)
        aux_flap.rotate(mfle, -m_path[mfpi,0]*np.pi/180)
        aux_flap.translate(m_path[mfpi,1:3]-mfle)

        section = afl.geo.Section([airfoil, main_flap, aux_flap])

section.scale([0,0], [0.9144, 0.9144])
section.rotate([0.9144/4,0], 0*np.pi/180)
# Inspect geometry
if True:
    section.plot(True)




md = afl.Mesh()
md.control_volume([-900, 1860], [-900, 900], True)
md.inflation_layer(60, 8.5, 190)
md.tangent_spacing(19, 0.84, 0.2, 0.5, 5, 50, [70,26])
md.normal_spacing(0.00002, 2, 1.16)
md.wake_region(70, [-10*np.pi/180, 10*np.pi/180], 78, 14)
t1 = time()
md.generate(section, 'naca23012_hld_RENAME', '.cgns', review=True)
print('Time elapsed: ' + str(time() - t1))





