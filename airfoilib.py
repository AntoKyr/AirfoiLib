# Copyright (c) 2025 Antonios-Ioakeim Kyriakopoulos


__version__ = '1.0.0'
__author__ = 'Antonios-Ioakeim Kyriakopoulos'
__license__ = 'ARR'  # until I read about open source licenses


import geomdl.BSpline, geomdl.NURBS, geomdl.fitting
import numpy as np
import geomdl
import gmsh
from matplotlib import pyplot
from matplotlib.path import Path
from copy import deepcopy
from typing import Callable
import csv
from tqdm import tqdm


class file:
    """
    Functions for reading and writting files.
    """
    @staticmethod
    def read_csv(name: str, te_round_if_open = False):
        """
        Read airfoil coordinates off of a csv file directly from airfoil tools and nowhere else (for now at least).

        Args:
            name: name of the file
        
        Returns:
            Airfoil object
        
        """
        if not name.endswith('.csv'):
            name+='.csv'
        with open(name) as datafile:
            reader = csv.reader(datafile)
            data =[]
            for row in reader:
                data.append(row)

        gaps = [i for i,line in enumerate(data) if line==['','']]
        ordinates = (data[gaps[0]+3 : gaps[1]])
        ordinates = list(map(lambda x: list(map(float, x)), ordinates))
        return Airfoil([ordinates], te_round_if_open)


class num:
    """
    Numerical analysis based functions.
    """
    opdict = {'<': lambda x,y: x<y, '>': lambda x,y: x>y, '<=': lambda x,y: x<=y, '>=': lambda x,y: x>=y}

    @staticmethod
    def bisect_solver(funct: Callable[[float], float], seg: list, tol: float) -> float:
        """
        Solve a function with the bisection method.

        Args:
            funct: function to be solved
            seg: [x1, x2] the starting segment
            tol: the maximum difference of x1 and x2 in the final segment,  tol > 0
        
        Returns:
            Estimated x root

        Raises:
            Error: Segment edges have common sign
            
        """
        y1, y2 = funct(seg[0]), funct(seg[1])
        while abs(seg[0] - seg[1]) > tol:
            midp = np.mean(seg)
            y0 = funct(midp)
            if y1*y0 < 0:
                y2, seg[1] = y0, midp
            elif y2*y0 < 0:
                y1, seg[0] = y0, midp
            elif y0 == 0:
                return midp
            else:
                raise Exception('Error: Segment edges have common sign')

        return (y2*seg[0] - y1*seg[1]) / (y2-y1)


    @staticmethod
    def chord_solver(funct: Callable[[float], float], seg: list, tol: float) -> float:
        """
        Solve a function with the chord method.

        Args:
            funct: function to be solved
            seg: [x1, x2] the starting segment
            tol: the maximum difference of x1 and x2 in the final segment,  tol > 0
        
        Returns:
            Estimated x root

        Raises:
            Error: Segment edges have common sign
            
        """
        y1, y2 = funct(seg[0]), funct(seg[1])
        midp = (y2*seg[0] - y1*seg[1]) / (y2-y1)
        dx = min(abs(seg[0]-midp), abs(seg[1]-midp))
        tol = tol/2

        while dx > tol:
            midp = (y2*seg[0] - y1*seg[1]) / (y2-y1)
            y0 = funct(midp)
            if y1*y0 < 0:
                dx = abs(seg[1] - midp)
                y2, seg[1] = y0, midp
            elif y2*y0 < 0:
                dx = abs(seg[0] - midp)
                y1, seg[0] = y0, midp
            elif y0 == 0:
                return midp
            else:
                raise Exception('Error: Segment edges have common sign')

        return (y2*seg[0] - y1*seg[1]) / (y2-y1)


    @staticmethod
    def basic_optim(funct: Callable[[float], float], seg: list, tol: float) -> float:
        """
        Maximize a function using a rudimentary routine.

        Args:
            funct: function to be optimized
            seg: [x1, x2] the starting segment
            tol: the maximum difference of x1 and x2 in the final segment,  tol > 0
        
        Returns:
            Estimated x optimum
            
        """
        x1, x2, x3 = seg[0], np.mean(seg), seg[1]
        y1, y2, y3 = funct(x1), funct(x2), funct(x3)

        while abs(x1 - x3) > tol:
            x12, x23 = (x1+x2)/2, (x2+x3)/2
            y12, y23 = funct(x12), funct(x23)
            if y12 > y23:
                x2, x3 = x12, x2
                y2, y3 = y12, y2
            else:
                x1, x2 = x2, x23
                y1, y2 = y2, y23

        x = [x1,x2,x3]
        y = [y1,y2,y3]

        return x[np.argmax(y)]


    @staticmethod
    def roll_average(y: list|tuple|np.ndarray, n: int) -> np.ndarray:
        """
        Calculate the rolling average of data values.

        Args:
            y: contains values
            n: range of average

        Returns:
            rolled data
        
        """
        if n == 0:
            return y
        y = np.array(y)
        r1 = range(n)
        r2 = range(len(y)-2*n, len(y)-n)
        csy = np.insert(np.cumsum(y),0,0)
        y2 = (csy[2*n+1:] - csy[:-2*n-1]) / (2*n+1)
        y1 = []
        y3 = []
        for k, i in enumerate(r1):
            y1.append(np.mean(y[0:i+k+1]))
        for k, i in enumerate(r2):
            y3.append(np.mean(y[i+k+1:]))

        return np.hstack((y1,y2,y3))


    @staticmethod
    def roots(x: list|tuple|np.ndarray, y: list|tuple|np.ndarray, mvt: float) -> np.ndarray:
        """
        Find the aproximate roots of discrete linearly interpolated data.

        Args:
            x: the x values of the data
            y: the y values of the data
            mvt: 0 < mvt < 1, the minimum variation tolerance, helps avoid finding multiple roots in noisy data, but lowers accuracy considerably

        Returns:
            the x roots
        
        """
        x, y = np.array(x), np.array(y)
        tol = mvt * max(np.abs(y))
        toli = np.nonzero(np.logical_not(np.logical_and(y < tol, y > -tol)))[0]
        x, y = x[toli], y[toli]
        rzi = np.nonzero(y[0:-1]*y[1:] == 0)[0]
        rni = np.nonzero(y[0:-1]*y[1:] < 0)[0]
        zroots = x[rzi]
        a = (y[rni+1] - y[rni]) / (x[rni+1] - x[rni])
        b = y[rni] - a * x[rni]
        nroots = - b/a
        return np.sort(np.hstack((zroots, nroots)))


    @staticmethod
    def derivative(x: list|tuple|np.ndarray, y: list|tuple|np.ndarray) -> np.ndarray:
        """
        Compute the linearly aproximated derivative of discrete linearly interpolated data.

        Args:
            x: the x values of the data
            y: the y values of the data

        Returns:
            the derivative values

        """
        x, y = np.array(x), np.array(y)
        return (y[1:] - y[0:-1]) / (x[1:] - x[0:-1])


class geo:
    """
    Geometric core of the library.
    """
    # BASIC TRANSFORMATIONS
    @staticmethod
    def translate(p: list|tuple|np.ndarray, tv: list|tuple|np.ndarray) -> np.ndarray:
        """
        Translate points by vector tv.

        Args:
            p: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
            tv: [x, y] components of displacement vector

        Returns:
            Translated point coordinates

        """
        p = np.array(p)
        tv = np.array(tv)
        if p.ndim == 1:
            return p + tv
        elif p.ndim ==2:
            return p + np.repeat([tv], np.shape(p)[0], axis=0)


    @staticmethod
    def rotate(p: list|tuple|np.ndarray, center: list|tuple|np.ndarray, theta: float) -> np.ndarray:
        """
        Rotate points p around center by theta.

        Args:
            p: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
            center: [x, y] coordinates of rotation center
            theta: the angle of rotation in radiants
        
        Returns:
            Rotated point coordinates

        """
        p = np.array(p)
        center = np.array(center)
        p = geo.translate(p, -center)
        transform = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        p = np.transpose(transform @ np.transpose(p))
        return geo.translate(p, center)


    @staticmethod
    def scale(p: list|tuple|np.ndarray, center: list|tuple|np.ndarray, fv: list|tuple|np.ndarray) -> np.ndarray:
        """
        Scale points p around center accodring to vector fv.

        Args:
            p: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
            center: [x, y] coordinates of rotation center
            fv: [xf, yf] factors by which each coordinate is scaled
        
        Returns:
            Scaled point coordinates

        """
        p = np.array(p)
        center = np.array(center)
        p = geo.translate(p, -center)
        p[:,0] = p[:,0] * fv[0]
        p[:,1] = p[:,1] * fv[1]
        return geo.translate(p, center)


    @staticmethod
    def mirror(p: list|tuple|np.ndarray, ax: list|tuple|np.ndarray) -> np.ndarray:
        """
        Mirror points p around axis ax.

        Args:
            p: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
            ax: [[xa1, ya1], [xa2, ya2]] a matrix of two points that define the mirroring axis 
        
        Returns:
            Mirrored point coordinates
        
        """
        p = np.array(p)
        ax = np.array(ax)
        theta = geo.vector.angle(ax[1]-ax[0])
        p = geo.translate(p, -ax[0])
        p = geo.rotate(p, [0,0], -theta)
        p[:,1] = - p[:,1]
        p = geo.rotate(p, [0,0], theta)
        p = geo.translate(p, ax[0])
        return p


    # LOCATION
    @staticmethod
    def inproxim(p1: list|tuple|np.ndarray, p2: list|tuple|np.ndarray, proxd: float) -> bool:
        """
        Check if two points are within proximity of each other.

        Args:
            p1: [x1, y1] coordinates of point 1
            p2: [x2, y2] coordinates of point 2
            proxd: proximity distance
        
        Returns:
            True if points are common
        
        """
        return proxd >= np.hypot(p1[0] - p2[0], p1[1] - p2[1])


    @staticmethod
    def inpolyg(p: list|tuple|np.ndarray, polyg: list|tuple|np.ndarray) -> np.ndarray:
        """
        Check if a point resides inside a polygon.
        
        Args:
            p: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
            polyg: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the polygons vertexes
        
        Return:
            contains booleans, True if point resides inside polyg

        """
        return Path(polyg).contains_points(p)


    @staticmethod
    def distance(p1: list|tuple|np.ndarray, p2: list|tuple|np.ndarray) -> np.ndarray:
        """
        Measure distance between two collections of points.

        Args:
            p1: [[x0, y0], [x1, y1], ... , [xm, ym]] the matrix containing all the point coordinates of collection 1
            p2: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of collection 2

        Returns:
            M x N matrix containing distances of all points, M and N being the number of points of collection 1 and 2

        """
        p1, p2 = np.array(p1), np.array(p2)
        m, n = np.shape(p1)[0], np.shape(p2)[0]
        p1 = np.repeat(p1[:,np.newaxis,:], n, axis=1)
        p2 = np.repeat(p2[np.newaxis,:,:], m, axis=0)
        p1x, p1y = p1[:,:,0], p1[:,:,1]
        p2x, p2y = p2[:,:,0], p2[:,:,1]
        return np.hypot(p1x-p2x, p1y-p2y)


    # SPLINE
    class spline:
        """
        A spline is the geomdl imported Curve class. Here are functions relating to splines.
        """
        @staticmethod
        def get_prmtr(nurbs: geomdl.BSpline.Curve, mode: str, val: list|tuple|np.ndarray, tol: float = 10**-3, delta: float = 10**-3) -> list:
            """
            Find the parameters of a spline.

            Args:
                nurbs: nurbs curve object
                mode: "linear" to attempt to find intersection parameters with line
                     | "length" to attempt to find parameter where the curve has certain length
                     | "point" to attempt to find parameter of a certain point
                     | "distance" to attempt to find a parameter where the curve has a certain distance from a point
                val: "linear" mode: 2 x N array containing line coefficients that describe the locus of the point to be interpolated
                     | "length" mode: contains the values of the lenght
                     | "point" mode: 2 x N array containing the point coordinates
                     | "distance" mode: 3 x N array containing the point coordinates from which the distance is measured, and the distance
                tol: the maximum difference of the values of the final bisection segment, the lower, the higher the accuracy
                delta: the curve discretization step, the lower, the higher the accuracy
            
            Returns:
                Contains all the found parameters.
                
            """
            delta = 1/(int(1/delta))
            if mode == "linear":
                if len(np.shape(val)) == 1:
                    val = [val]
                tol = tol * delta
                nurbs.delta =  1/(1/delta + 1)
                crv = np.array(nurbs.evalpts)
                params = []
                for pv in val:
                    ps = crv[:, 1] - np.polyval(pv, crv[:, 0])
                    # Find all segments
                    sci = np.array(np.nonzero(ps[0:-1]*ps[1:] < 0))[0]

                    # Find parameters
                    u = []
                    def funct(u):
                        evp = nurbs.evaluate_single(u)
                        return evp[1] - np.polyval(pv, evp[0])

                    for si in sci:
                        seg = [si * delta, (si+1) * delta]
                        u.append(num.chord_solver(funct, seg, tol))

                    params = params + u

            if mode == "length":
                if len(np.shape(val)) == 0:
                    val = [val]
                tol = tol * delta
                nurbs.delta =  1/(1/delta + 1)
                crv = np.array(nurbs.evalpts)
                crvlen = geo.curve.length(crv)
                params = []
                for pv in val:
                    if pv == 0:
                        params.append(0)
                        continue
                    clen = crvlen - pv
                    # Find segment
                    si = np.array(np.nonzero(clen > 0))[0]
                    if len(si) == 0:
                        print('geo.spline.get_prmtr Warning: Requested length of '+str(pv)+' exceeds curve length of '+str(clen[-1]+pv)+', continueing to next value.')
                        continue
                    si = si[0]
                    csi = crv[si-1]
                    seg = [(si-1) * delta, si * delta]
                    # Find parameter
                    def funct(u):
                        p = nurbs.evaluate_single(u)
                        return np.hypot(p[0] - csi[0], p[1] - csi[1]) + clen[si-1]

                    u = num.chord_solver(funct, seg, tol)
                    params.append(u)
            
            if mode == "point":
                if len(np.shape(val)) == 1:
                    val = [val]
                tol = tol * delta
                nurbs.delta =  1/(1/delta + 1)
                crv = np.array(nurbs.evalpts)
                params = []
                dists = geo.distance(val, crv)
                for i,pv in enumerate(val):
                    # Find segment
                    seg = np.sort(np.argsort(dists[i])[0:2])*delta
                    
                    def funct(u):
                        p = nurbs.evaluate_single(u)
                        return -(p[0]-pv[0])**2 -(p[1]-pv[1])**2
                    
                    u = num.basic_optim(funct, seg, tol)
                    params.append(u)
            
            if mode == "distance":
                if len(np.shape(val)) == 1:
                    val = [val]
                tol = tol * delta
                nurbs.delta =  1/(1/delta + 1)
                crv = np.array(nurbs.evalpts)
                params = []
                for pv in val:
                    cd = geo.distance([pv[0:2]], crv)[0] - pv[2]
                    # Find all segments
                    sci = np.array(np.nonzero(cd[0:-1]*cd[1:] < 0))[0]
                    # Find parameters
                    u = []
                    def funct(u):
                        return geo.distance([nurbs.evaluate_single(u)], [pv[0:2]])[0,0] - pv[2]

                    for si in sci:
                        seg = [si * delta, (si+1) * delta]
                        u.append(num.chord_solver(funct, seg, tol))

                    params = params + u
                
            return params


        @staticmethod
        def bezier(p: list|tuple|np.ndarray, w: list|tuple|np.ndarray = 1) -> geomdl.NURBS.Curve:
            """
            Get a weighted bezier curve.

            Args:
                p: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the control point coordinates
                w: weights
            
            Returns:
                Curve object

            """
            if type(p) == np.ndarray:
                p = p.tolist()
            if w == 1:
                w = np.ones(len(p))
            pl = len(p)
            degree = pl-1
            knots = np.zeros(pl).tolist() + np.ones(pl).tolist()
            bcrv = geomdl.NURBS.Curve()
            bcrv.degree = degree
            bcrv.ctrlpts = p
            bcrv.weights = w
            bcrv.knotvector = knots
            return bcrv


        @staticmethod
        def fit(p: list|tuple|np.ndarray, degree: int = 3) -> geomdl.BSpline.Curve:
            """
            Fit a spline onto points.

            Args:
                p: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
                degree: the piecewise polynomial degree, will be ignored if there arent enough points
            
            Returns:
                Curve object
                
            """
            if type(p) == np.ndarray:
                p = p.tolist()
            if len(p) < degree + 1:
                degree = len(p) - 1
            return geomdl.fitting.interpolate_curve(p, degree, centripetal=True)


        @staticmethod
        def length(nurbs: geomdl.BSpline.Curve, param: list|tuple|np.ndarray, delta: float = 10**-4) -> np.ndarray:
            """
            Find the aproximate length of a spline curve at given parameters.

            Args:
                nurbs: nurbs curve object 
                param: parameters at which to evaluate length
                delta: the curve discretization step, the lower, the higher the accuracy

            Returns:
                Contains all the length values of the parameters
            
            """
            dv = np.linspace(0,1, int(1/delta))
            dv = np.sort(np.hstack((dv, param)))
            evali = np.searchsorted(dv, param) + np.arange(len(param))
            return geo.curve.length(nurbs.evaluate_list(dv))[evali]


    # VECTOR
    class vector:
        """
        A vector is an array with size 2, containing the vector's components. Here are function relating to vectors.
        """
        quadict = {1: [1,1], 2: [-1,1], 3: [-1,-1], 4:[1,-1]}

        @staticmethod
        def unit(v):
            """
            Get the unit vector of the vector.

            Args:
                v: vector

            Returns:
                unit vector

            """
            v = np.array(v)
            if v.ndim == 2:
                norm = np.hypot(v[:,0], v[:,1])
                norm = np.transpose([norm, norm])
            elif v.ndim == 1:
                norm = np.hypot(v[0], v[1])
            return v/norm


        @staticmethod
        def angle(v1: list|tuple|np.ndarray, v2: list|tuple|np.ndarray = [1,0], reflex: bool = False) -> float:
            """
            Calculate the angle of a vector and x axis, or between two vectors, from vector 2 to vector 1.

            Args:
                v1: [x, y] coordinates of vector 1
                v2: [x, y] coordinates of vector 2
                reflex: If True, the opposite angle is given 
            
            Returns:
                Angle in radiants

            """
            v1, v2 = np.array(v1), np.array(v2)
            if v1.ndim == 1:
                normprod = np.linalg.norm(v1) * np.linalg.norm(v2)
                costh = np.dot(v2, v1) / normprod
                v1, v2 = np.append(v1,0), np.append(v2,0)
                sinth = np.cross(v2, v1)[2] / normprod
            else:
                if v2.ndim == 1:
                    v2 = np.hstack((np.ones((len(v1),1)), np.zeros((len(v1),1))))
                
                normprod = np.hypot(v1[:,0], v1[:,1]) * np.hypot(v2[:,0], v2[:,1])
                costh = np.diagonal(v2 @ np.transpose(v1)) / normprod
                v1, v2 = np.hstack((v1, np.zeros((len(v1),1)))), np.hstack((v2, np.zeros((len(v1),1))))
                sinth = np.cross(v2, v1)[:,2] / normprod

            theta = np.arctan2(sinth, costh)
            if reflex:
                theta = theta - np.sign(theta) * 2*np.pi
            return theta


        @staticmethod
        def quadrant(v1: list|tuple|np.ndarray, v2: list|tuple|np.ndarray = np.zeros(2), reflex: bool = False) -> int:
                """
                Find the quadrant at which a vector, or the bisector of two vectors, point at.
                
                Args:
                    v1: [x, y] coordinates of vector 1
                    v2: [x, y] coordinates of vector 2
                    reflex: If true, the bisector of the reflex angle is taken instead

                Returns:
                    The number of the quadrant        

                """
                if reflex:
                    s = -1
                else:
                    s = 1
                v1 = geo.vector.unit(v1)
                if not np.all(v2 == np.zeros(2)):
                    v2 = geo.vector.unit(v2)
                v = (v1 + v2) * s
                if v[0] >= 0 and v[1] > 0:
                    return 1
                elif v[0] < 0 and v[1] >= 0:
                    return 2
                elif v[0] <= 0 and v[1] < 0:
                    return 3
                elif v[0] > 0 and v[1] <= 0:
                    return 4


        @staticmethod
        def bisector(v1: list|tuple|np.ndarray, v2: list|tuple|np.ndarray) -> np.ndarray:
            """
            Find the unit vector that bisects two vectors.

            Args:
                v1: [x, y] coordinates of vector 1
                v2: [x, y] coordinates of vector 2
            
            Returns:
                [x, y] coordinates of bisectorvector

            """
            v = geo.vector.unit(v1) + geo.vector.unit(v2)
            return geo.vector.unit(v)


        @staticmethod
        def vertical(v: list|tuple|np.ndarray) -> np.ndarray:
            """
            Find a vector vertical to the given.

            Args:
                v: [x, y] coordinates of vector
                side: If True the right side is picked, esle, the left
            
            Returns:
                [x, y] coordinates of vertical vector

            """
            a = [[v[0], v[1]], [-v[1], v[0]]]
            b = [0, 1]
            vv = np.linalg.solve(a,b)
            return geo.vector.unit(vv)


    # LINE
    class line:
        """
        A line is an array with size 2, containing its coefficients. Here are function relating to lines.
        """
        @staticmethod
        def bisector(lf1: list|tuple|np.ndarray, lf2: list|tuple|np.ndarray) -> list:
            """
            Find the line bisectors of two lines.

            Args:
                lf1: Line 1 factors as given by np.polyfit()
                lf2: Line 2 factors as given by np.polyfit()
            
            Returns:
                list containing:
                -lfb1 (ndarray): Bisector line 1 factors as given by np.polyfit()
                -lfb2 (ndarray): Bisector line 2 factors as given by np.polyfit()
            
            """
            sqr1 = (lf1[0]**2 + 1)**0.5
            sqr2 = (lf2[0]**2 + 1)**0.5
            lfb1 = np.zeros(2)
            lfb1[0] = (lf1[0] * sqr2 + lf2[0] * sqr1) / (sqr2 + sqr1)
            lfb1[1] = (lf1[1] * sqr2 + lf2[1] * sqr1) / (sqr2 + sqr1)
            lfb2 = np.zeros(2)
            lfb2[0] = (lf1[0] * sqr2 - lf2[0] * sqr1) / (sqr2 - sqr1)
            lfb2[1] = (lf1[1] * sqr2 - lf2[1] * sqr1) / (sqr2 - sqr1)
            return [lfb1, lfb2]


        @staticmethod
        def vertical(lf: list|tuple|np.ndarray, p: list|tuple|np.ndarray) -> np.ndarray:
            """
            Find the line vertical to the first, passing through point.

            Args:
                lf: Line factors as given by np.polyfit()
                p: [x, y] coordinates of point
            
            Returns:
                The vertical line factors as given by np.polyfit()
            
            """
            lfv = np.zeros(2)
            lfv[0] = -1/lf[0]
            lfv[1] = p[1] - lfv[0]*p[0]
            return list(lfv)


        @staticmethod
        def intersect(lf1: list|tuple|np.ndarray, lf2: list|tuple|np.ndarray) -> np.ndarray:
            """
            Find the intersection of two lines.

            Args:
                lf1: Line 1 factors as given by np.polyfit()
                lf2: Line 2 factors as given by np.polyfit()
            
            Returns:
                [x, y] coordinates of the intersection
            
            """
            x0 = (lf2[1]-lf1[1])/(lf1[0]-lf2[0])
            y0 = (lf1[0]*lf2[1] - lf2[0]*lf1[1])/(lf1[0]-lf2[0])
            return np.array([x0, y0])


        @staticmethod
        def project(p: list|tuple|np.ndarray, lf: list|tuple|np.ndarray) -> np.ndarray:
            """
            Project a point onto a line.

            Args:
                p: [x, y] coordinates of point
                lf: Line factors as given by np.polyfit()

            Returns:
                [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the projected point coordinates

            """
            lfv = np.zeros(2)
            lfv[0] = -1/lf[0]
            lfv[1] = p[1] - lfv[0]*p[0]
            return geo.line.intersect(lf, lfv)


        @staticmethod
        def fit(p1: list|tuple|np.ndarray, p2: list|tuple|np.ndarray) -> np.ndarray:
            """
            Get the polynomial factors of the line passing through two points.

            Args:
                p1: [x0, y0] the matrix containing the point 1 coordinates
                p2: [x0, y0] the matrix containing the point 2 coordinates

            Returns:
                [a1, a0] the line poly factors

            """
            x1, y1 = p1
            x2, y2 = p2
            return np.array([(y2 - y1) / (x2 - x1), (y1 * x2 - y2 * x1) / (x2 - x1)])


        @staticmethod
        def angle(lf: list|tuple|np.ndarray) -> float:
            """
            Get the angle of a line segment from x axis.

            Args:
                lf: Line factors as given by np.polyfit()

            Returns:
                angle
                
            """
            return np.atan2(lf[0])


    # CIRCLE
    class circle:
        """
        Here are functions relating to circles.
        """
        @staticmethod
        def fit2pr(p1: list|tuple|np.ndarray, p2: list|tuple|np.ndarray, r: float, side:bool = True) -> np.ndarray:
            """
            Find a circle passing through two points, with a given radius.

            Args:
                p1: [x, y] coordinates of point 1
                p2: [x, y] coordinates of point 2
                r: radius of the circle
                side: If true the center at the right side of the vector (p2 -p1) is returned. Else, the left.
            
            Returns:
                [x, y] coordinates of the center of the circle
            
            """
            p1, p2 = np.array(p1), np.array(p2)
            b = (p2[1]**2 - p1[1]**2 + p2[0]**2 - p1[0]**2) / (p2[0] - p1[0])
            a = (p1[1] - p2[1]) / (p2[0] - p1[0])
            a2 = 1 + a**2
            a1 = b*a - 2*p1[1] - 2*p1[0]*a
            a0 = (b/2)**2 - p1[0]*b + p1[0]**2 + p1[1]**2 - r**2
            yc = np.roots([a2, a1, a0])
            xc = b/2 + a*yc
            centers = np.transpose([xc, yc])
            v1 = centers[0] - p1
            v2 = p2 - p1

            if side:
                s = 1
            else:
                s = -1

            if s * np.cross(v1, v2) > 0:
                return centers[0]
            else:
                return centers[1]


        @staticmethod
        def fit(p: list|tuple|np.ndarray) -> list:
            """
            Fit a circle onto points. Points must be 3 or more.

            Args:
                p: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
            
            Returns:
                list containing:
                - p0 (ndarray): the coordinates of the circle center
                - r (float): the circle radius
            
            """
            p = np.array(p)
            n = np.shape(p)[0]
            x, y = np.sum(p[:,0]), np.sum(p[:,1])
            x2, y2 = np.sum(p[:,0]**2), np.sum(p[:,1]**2)
            xy = np.sum(p[:,0] * p[:,1])
            a = [[2*x2, 2*xy, x], [2*xy, 2*y2, y], [2*x, 2*y, n]]
            b = [np.sum(p[:,0]**3 + p[:,0] * p[:,1]**2), np.sum(p[:,1]**3 + p[:,1] * p[:,0]**2), x2 + y2]
            sv = np.linalg.solve(a, b)
            c = (1/n) * (x2 + y2 - 2 * np.dot(sv, [x, y, 0]))
            r = (sv[0]**2 + sv[1]**2 + c)**0.5
            p0 = np.array([sv[0], sv[1]])
            return [p0, r]


        @staticmethod
        def tang2seg(seg1: list|tuple|np.ndarray, seg2: list|tuple|np.ndarray) -> np.ndarray:
            """
            Find the minimum and maximum radius of a circle tangent to two line segments, in the quadrant pointed at by their bisecting vector.

            Args:
                seg1: [[x0, y0], [x1, y1]] the point coordinates of segment 1
                seg2: [[x0, y0], [x1, y1]] the point coordinates of segment 2
            
            Returns:
                list containing:
                -minimum and maximun radius of the circle
                -empty list if no tangent circle exists
            
            """
            seg1, seg2 = np.array(seg1), np.array(seg2)
            lf1 = geo.line.fit(seg1[0], seg1[1])
            lf2 =  geo.line.fit(seg2[0], seg2[1])
            p0 = geo.line.intersect(lf1, lf2)
            insidebool1 = min(seg1[:,0]) <= p0[0] <= max(seg1[:,0])      # Is intersection point inside segment?
            insidebool2 = min(seg2[:,0]) <= p0[0] <= max(seg2[:,0])      # Is intersection point inside segment?
            pointawaybool1 = np.linalg.norm(p0-seg1[0]) < np.linalg.norm(p0-seg1[1])  # Is segment vector pointing away from intersection point?
            pointawaybool2 = np.linalg.norm(p0-seg2[0]) < np.linalg.norm(p0-seg2[1])  # Is segment vector pointing away from intersection point?

            if ((not insidebool1) and (not pointawaybool1)) or ((not insidebool2) and (not pointawaybool2)):
                return []

            bisectvct = geo.vector.bisector(seg1[1]-seg1[0], seg2[1]-seg2[0])
            if np.any(bisectvct == np.nan):
                return []
            theta = geo.vector.angle(bisectvct)
            blf = geo.line.fit(p0, bisectvct + p0)
            p11 = geo.line.intersect(geo.line.vertical(lf1, seg1[0]), blf)
            p12 = geo.line.intersect(geo.line.vertical(lf1, seg1[1]), blf)
            p21 = geo.line.intersect(geo.line.vertical(lf2, seg2[0]), blf)
            p22 = geo.line.intersect(geo.line.vertical(lf2, seg2[1]), blf)
            segpoints = np.array([p11,p12,p21,p22])
            intersi = list(np.argsort(geo.rotate(segpoints, [0,0], -theta)[:,0]))[1:-1]
            if intersi == [3,0] or intersi == [1,2]:                   # No intersecting bisector segments
                return []
            interseg = segpoints[intersi]
            clp = np.vstack((seg1, seg2))[intersi]
            r = np.sort([np.linalg.norm(clp[0]-interseg[0]), np.linalg.norm(clp[1]-interseg[1])])
            if insidebool1 and insidebool2:                            # If segments intersect minimum radius is set to 0
                r[0] = 0
                interseg[0] = p0
         
            return r


        @staticmethod
        def tang2lnr(lf1: list|tuple|np.ndarray, lf2: list|tuple|np.ndarray, r: float, quadrnt: int) -> list:
            """
            Find circle tangent to two lines with radius r.

            Args:
                lf1: Line 1 factors as given by np.polyfit()
                lf2: Line 2 factors as given by np.polyfit()
                r: Radius of the circle
                quadrnt: The quadrant that the center is located, in relation to the intersection of lines 1 and 2
            
            Returns:
                list containing:
                -p0 (ndarray): [x, y] coordinates of the center of the circle
                -ptan1 (ndarray): [x, y] coordinates of point of tangency on line 1
                -ptan2 (ndarray): [x, y] coordinates of point of tangency on line 2
            
            """
            # Find centers
            lf11 = [lf1[0], lf1[1] + (lf1[0]**2 + 1)**0.5 * r]
            lf12 = [lf1[0], lf1[1] - (lf1[0]**2 + 1)**0.5 * r]
            lf21 = [lf2[0], lf2[1] + (lf2[0]**2 + 1)**0.5 * r]
            lf22 = [lf2[0], lf2[1] - (lf2[0]**2 + 1)**0.5 * r]
            p0 = np.array([geo.line.intersect(lf11, lf21), geo.line.intersect(lf11, lf22), geo.line.intersect(lf12, lf21), geo.line.intersect(lf12, lf22)])

            # Select center
            xp, yp = geo.line.intersect(lf1, lf2)
            fx, fy = geo.vector.quadict[quadrnt]
            i = np.argwhere(np.logical_and(fx * (p0[:,0] - xp) >= 0, fy * (p0[:,1] - yp) >= 0))[0,0]
            p0 = p0[i]

            # Find tangent points
            lfc = geo.line.vertical(lf1, p0)
            ptan1 = geo.line.intersect(lfc, lf1)
            lfc = geo.line.vertical(lf2, p0)
            ptan2 = geo.line.intersect(lfc, lf2)

            return [p0, ptan1, ptan2]


        @staticmethod
        def tang2pln(lf: list|tuple|np.ndarray, p1: list|tuple|np.ndarray, p2: list|tuple|np.ndarray) -> np.ndarray:
            """
            Find the circle tangent on a line on point 1 and passing through point 2.

            Args:
                lf: Line factors as given by np.polyfit()
                p1: [x, y] coordinates of point 1, the coordinates must satisfy the line equation
                p2: [x, y] coordinates of point 2
            
            Returns:
                [x, y] coordinates of the center of the circle

            """
            # Find intersecting lines
            lf1 = np.zeros(2)
            lf1[0] = -1/lf[0]
            lf1[1] = p1[1] - lf1[0] * p1[0]
            lf2 = geo.line.fit(p1, p2)
            lf2[0] = -1/lf2[0]
            lf2[1] = (p1[1] + p2[1]) / 2 - lf2[0] * (p1[0] + p2[0]) / 2
            # Return center
            return geo.line.intersect(lf1, lf2)


        @staticmethod
        def tang2crv(c: list|tuple|np.ndarray, p0: list|tuple|np.ndarray, r: float) -> list:
            """
            Find the circle tangent to a curve's segments, with center p0 and the closest possible matching radius to the requested

            Args:
                c: curve
                p0: center of the circle
                r: radius

            Returns:
                list containing:
                    -i: the index of the best tangential segment
                    -ptan: the point of tangency
                    -rf: the radius found
                    - empty list if no tangent is found

            """
            c = np.array(c)
            i, ptan, rf = [], [], []
            for j in range(len(c)-1):
                pproj = geo.line.project(p0, geo.line.fit(c[j], c[j+1]))
                xlims = np.sort(c[j:j+2,0])
                if xlims[0] <= pproj[0] <= xlims[1]:
                    rf.append(np.hypot(p0[0]-pproj[0], p0[1]-pproj[1]))
                    ptan.append(pproj)
                    i.append(j)
            dr = np.abs(np.array(rf) - r)
            k = np.argmin(dr)
            return [i[k], ptan[k], rf[k]]


        @staticmethod
        def arc(p1: list|tuple|np.ndarray, p2: list|tuple|np.ndarray, p0: list|tuple|np.ndarray, n: int, reflex: bool = False) -> np.ndarray:
            """
            Generate points of an arc, from point 1 to point 2 with point 0 as center. Generated points include point 1 and 2.

            Args:
                p1: [x, y] coordinates of point 1
                p2: [x, y] coordinates of point 2
                p0: [x, y] coordinates of the center
                n: Number of points to be generated
                reflex: If False, the smaller of the two possible arcs will be generated, else the greater
            
            Returns:
                [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the arc

            """
            p1, p2, p0 = np.array(p1), np.array(p2), np.array(p0)
            r = np.linalg.norm(p1-p0)
            theta = geo.vector.angle(p2-p0, p1-p0, reflex)
            theta1 = geo.vector.angle(p1-p0)
            thetan = np.linspace(theta1, theta1 + theta, n + 1)[1:-1]
            x = r * np.cos(thetan) + p0[0]
            y = r * np.sin(thetan) + p0[1]
            return np.vstack((p1, np.transpose([x, y]), p2))


    # CURVE
    class curve:
        """
        A curve is a N x 2 array, a collection of point coordinates through which a curve passes through. Here are functions for curves.
        """
        # Metrics
        @staticmethod
        def segment_len(c: list|tuple|np.ndarray, relative: bool = False) -> np.ndarray:
            """
            Calculate the length of each segment of the curve.

            Args:
                c: curve
                relative: if True return the relative length of each segment

            Returns:
                vector containing the length of each segment

            """
            c = np.array(c)
            norms = np.hypot(c[1:,0] - c[0:-1,0], c[1:,1] - c[0:-1,1])
            if relative:
                norms = norms / sum(norms)
            return norms


        @staticmethod
        def length(c: list|tuple|np.ndarray, relative: bool = False) -> np.ndarray:
            """
            Calculate the length of the curve on every point.

            Args:
                c: curve
                relative: if True return the relative length (0 -> 1)

            Returns:
                vector containing the length of the curve up to every point of it

            """
            c = np.array(c)
            norms = geo.curve.segment_len(c)
            clen = np.cumsum(norms)
            clen = np.insert(clen, 0, 0)
            if relative:
                clen = clen / clen[-1]
            return clen


        @staticmethod
        def curvature(c: list|tuple|np.ndarray) -> np.ndarray:
            """
            Calculate the curvature of the curve.

            Args:
                c: curve

            Returns:
                vector containing the curvature value for every point of the curve

            """
            c = np.array(c)
            if len(c) > 2:
                cang = geo.vector.angle(c[2:] - c[1:-1], c[1:-1] - c[0:-2])
                cang = np.append(np.insert(cang, 0, cang[0]), cang[-1])
                slen = geo.curve.segment_len(c)
                divlen = np.min([slen[0:-1], slen[1:]], axis=0)
                divlen = np.append(np.insert(divlen, 0, slen[0]), slen[-1])
                return cang / divlen
            else: 
                return np.array([0,0])


        @staticmethod
        def normals(c: list|tuple|np.ndarray) -> np.ndarray:
            """
            Calculate the normals for every point of the curve.

            Args:
                c: curve
            
            Returns:
                an array with all the vectors normal to the curve.

            """
            c = np.array(c)
            # edge normals
            v_0 = geo.vector.vertical(c[0]-c[1])
            v_1 = geo.vector.vertical(c[-2]-c[-1])
            if len(c) > 2:
                v1 = c[2:] - c[1:-1]
                v2 = c[0:-2] - c[1:-1]
                angs = geo.vector.angle(v1, v2)
                nega = np.nonzero(angs<0)[0]
                angs[nega] = angs[nega] + 2*np.pi
                angs = angs/2
                cosa, sina = np.cos(angs), np.sin(angs)
                v2 = geo.vector.unit(v2)
                xn = v2[:,0]*cosa - v2[:,1]*sina
                yn = v2[:,0]*sina - v2[:,1]*cosa
                return np.vstack((v_0, np.transpose([xn,yn]), v_1))
            else:
                return np.vstack((v_0, v_1))


        @staticmethod
        def plot(c: list|tuple|np.ndarray, show: bool = False, **plotargs):
            """
            Plot the curve.

            Args:
                c: curve
                plotargs: arguments to pass to the pyplot.plot function
            
            """
            c = np.array(c)
            pyplot.plot(c[:,0], c[:,1], **plotargs)
            if show:
                pyplot.grid()
                pyplot.axis('equal')
                pyplot.show()


        @staticmethod
        def interp(c: list|tuple|np.ndarray, mode: str, val: list|tuple|np.ndarray, repac: float = 4, tol = 0.01) -> list:
            """
            Interpolate a curve using a 3rd degree spline.

            Args:
                c: curve
                mode: interpolation mode, see: geo.spline.get_prmtr
                val: contains the values depending on mode, see: geo.spline.get_prmtr
                repac: representation accuracy, the higher this is, the higher the accuracy of the function, depending on the accuracy that the curve points repressent itself
                tol: geo.spline.get_prmtr tolerance

            Returns:
                Contains all the found points.
                
            """
            spl = geo.spline.fit(c)
            delta = 1 / (repac/min(geo.curve.segment_len(c, relative=True)) + 1)
            params = list(geo.spline.get_prmtr(spl, mode, val, tol, delta))
            return np.array(spl.evaluate_list(params))


        @staticmethod
        def loop(c: list|tuple|np.ndarray) -> list:
            """
            Find the first loop of a curve, meaning the first intersection with itself.

            Args:
                c: curve
            
            Returns:
                list containing:
                -p (ndarray): [x, y] coordinates of the intersection
                -i (int): index of coordinates of last point before the first segment of intersection
                -j (int): index of coordinates of last point before the second segment of intersection
            
            """
            c = np.array(c)
            for ic in range(len(c)-1):
                inters = geo.curve.intersect(c[ic:ic+2], c[ic+2:], True)
                if len(inters) > 0:
                    return inters[0]
            return []


        @staticmethod
        def draw(fignum = 1, axis = [-10,110, -60, 60]) -> list:
            """
            Draw curves by adding points, within a matplotlib figure.

            Args:
                fignum: the figure at which to draw
            
            Returns:
                list containing curves

            """
            crv = []
            crvl = []
            fig = pyplot.figure(fignum)
            pyplot.axis(axis)
            pyplot.title('Draw curves')
            pyplot.xlabel('[ Click: place point | Enter: finish curve | Escape: exit and return]')
            pyplot.grid()

            def place_point(event):
                x, y = event.xdata, event.ydata
                pyplot.plot(x, y, '.r')
                if len(crv) > 0:
                    pyplot.plot([crv[-1][0], x], [crv[-1][1], y],'b')
                crv.append([x, y])
                pyplot.draw()

            def keyvent(event):
                if event.key == 'enter':
                    crvl.append(np.array(crv))
                    crv.clear()
                elif event.key == 'escape':
                    fig.canvas.mpl_disconnect(cid2)
                    fig.canvas.mpl_disconnect(cid)
                    pyplot.close()

            cid2 = fig.canvas.mpl_connect('key_press_event', keyvent)
            cid = fig.canvas.mpl_connect('button_press_event', place_point)

            pyplot.show()
            return crvl


        @staticmethod
        def proxi2p(c: list|tuple|np.ndarray, p: list|tuple|np.ndarray) -> list:
            """
            Get the closest point of the curve's segment to the point given.

            Args:
                c: curve
                p: [x, y] point coordinates to find the closest curve point to
            
            Returns:
                list containing:
                    -i: the index of the closest point or segment that contains it
                    -p0: the closest point coordinates
            
            """
            c = np.array(c)
            i, p0, df = [], [], []
            for j in range(len(c)-1):
                pproj = geo.line.project(p, geo.line.fit(c[j], c[j+1]))
                xlims = np.sort(c[j:j+2,0])
                if xlims[0] <= pproj[0] <= xlims[1]:
                    df.append(np.hypot(p[0]-pproj[0], p[1]-pproj[1]))
                    p0.append(pproj)
                    i.append(j)
            df += geo.distance([p], c)[0].tolist()
            p0 += c.tolist()
            i += list(range(len(c)))
            k = np.argmin(df)
            return [i[k], p0[k]]


        # Morphology
        @staticmethod
        def addpoints(c: list|tuple|np.ndarray, p: list|tuple|np.ndarray, ends: list|tuple|np.ndarray) -> np.ndarray:
            """
            Add points to a curves ends.

            Args:
                c: curve
                p: array of point coordinates to be added
                ends: array of integers indicating the end that they will be added, 0 for the beginning of the curve, 1 for the end of it
            
            Returns:
                curve with added points
            
            """
            for i,e in enumerate(ends):
                if e == 1:
                    c = np.append(c, [p[i]], axis=0)
                elif e == 0:
                    c = np.insert(c, 0, p[i], axis=0)
            return c


        @staticmethod
        def offset(c: list|tuple|np.ndarray, d: float):
            """
            Create a parallel curve.

            Args:
                c: curve
                d: the multipler of the offset

            Returns:
                offset curve

            """
            c = np.array(c)
            return c + geo.curve.normals(c) * d


        @staticmethod
        def fit2p(c: list|tuple|np.ndarray, p1: list|tuple|np.ndarray, p2: list|tuple|np.ndarray, i1: int = 0, i2: int = -1, proxi_snap: bool = False) -> np.ndarray:
            """
            Transforms points of a curve, so the indexed points coincide with given coordinates.

            Args:
                c: curve
                p1: [x, y] coordinates of point 1
                p2: [x, y] coordinates of point 2
                i1: index of point of curve that will be moved onto point 1
                i2: index of point of curve that will be moved onto point 2
                proxi_snap: if True, i1 and i2 are ignored and instead the points of the curve closest to the points 1 and 2 will be selected accordingly.
            
            Returns:
                transformed point coordinates

            """
            c = np.array(c)
            p1 = np.array(p1)
            p2 = np.array(p2)

            if proxi_snap:
                dists = geo.distance([p1,p2], c)
                indxs = np.argmin(dists, axis=1)
                if indxs[0] == indxs[1]:
                    print('geo.curve.fit2p Warning: crv_fit2p proxi_snap failed, defaulting to the use of index arguments.')
                else:
                    i1, i2 = indxs[0], indxs[1]

            c = geo.translate(c, p1-c[i1])
            theta = geo.vector.angle(c[i2] - c[i1], p2 - p1)
            c = geo.rotate(c, p1, -theta)
            sf = np.linalg.norm(p2 - p1) / np.linalg.norm(c[i2] - c[i1])
            c = geo.scale(c, p1, [sf, sf])
            return c


        @staticmethod
        def snap(c: list|tuple|np.ndarray, coords: list|tuple|np.ndarray) -> np.ndarray:
            """
            Snap a curve's end onto given coordinates so it retains its original general shape. (The curve's start remains at the original point)

            Args:
                c: curve
                coords: [x, y] coordinates to snap to

            Returns:
                snapped curve

            """
            c = np.array(c)
            relen = geo.curve.length(c,True)
            relen = np.transpose([relen,relen])
            cf = geo.curve.fit2p(c, c[0], coords)
            return cf * relen + (1 - relen) * c


        @staticmethod
        def bend(c: list|tuple|np.ndarray, theta: float) -> np.ndarray:
            """
            Incrementally rotate all curve segments around their first point.

            Args:
                c: curve
                theta: bending angle

            Returns:
                bent curve

            """
            c = np.array(c)
            rotang = geo.curve.segment_len(c, True) * theta
            for i in range(1,len(c)):
                c[i:] = geo.rotate(c[i:], c[i-1], rotang[i-1])
            return c


        @staticmethod
        def snip(c: list|tuple|np.ndarray, mode: str, val: list|tuple|np.ndarray, fid: int, repac: float = 4, tol = 0.01) -> list:
            """
            Split a curve into seperate curves at particular paramater values of the spline that passes through it. The curve must be well repressented for this to work properly.
            In order to make this robust, it has become very resourve intensive, so use with care.

            Args:
                c: curve
                mode: interpolation mode, see: geo.spline.get_prmtr
                val: contains the values depending on mode, see: geo.spline.get_prmtr
                fid: the greater the fid, the more accurate the segments are to the original curve
                repac: representation accuracy, the higher this is, the higher the accuracy of the function, depending on the accuracy that the curve repressents itself
                tol: geo.spline.get_prmtr tolerance

            Returns:
                list of curve segments
            
            """
            if len(np.shape(val)) == 1:
                if len(val) == 0:
                    return [c]
            c = np.array(c)
            spl = geo.spline.fit(c)            
            seglen = geo.curve.segment_len(c)
            clen = np.insert(np.cumsum(seglen),0,0)
            delta = 1 / (repac/min(seglen/clen[-1]) + 1)
            params = geo.spline.get_prmtr(spl, mode, val, tol, delta)
            if len(params) == 0:
                return [c]
            params.sort()

            # find next and previous points
            lc = len(c)-1
            maxsegi = np.argmax(seglen)
            # get maximum parameter difference
            du = geo.spline.get_prmtr(spl, 'length', clen[maxsegi:maxsegi+2], 10**-2, delta)
            du = du[1] - du[0]
            valen = geo.spline.length(spl, params, delta)
            dists = geo.distance(spl.evaluate_list(params), c)
            proxi = np.argsort(dists, axis=1)

            ppar, npar, snipi = [], [], []
            for i in range(len(dists)):
                # limit the paramaters of closest points
                ceili = np.nonzero(clen > valen[i])[0]
                if len(ceili) > 1:
                    ceili = ceili[0]
                    ceilu = params[i] + (1 + repac)*du
                else:
                    ceilu = 1.1
                floori = np.nonzero(clen < valen[i])[0]
                if len(floori) > 1:
                    floori = floori[-1]
                    flooru = params[i] - (1 + repac)*du
                else:
                    flooru = -0.1

                # search nearest points
                found1, found2 = False, False
                j = 0
                while not (found1 and found2):

                    if proxi[i,j] == 0:
                        proxipar = 0
                    elif proxi[i,j] == lc:
                        proxipar = 1
                    else: 
                        proxipar = geo.spline.get_prmtr(spl,'point', c[proxi[i,j]], tol, delta)[0]
                    
                    if (flooru < proxipar < params[i]) and (not found1):
                        prev_param = proxipar
                        snipi.append(proxi[i,j])
                        found1 = True
                    elif (ceilu > proxipar > params[i]) and (not found2):
                        next_param = proxipar
                        found2 = True
                    j += 1

                ppar.append(prev_param)
                npar.append(next_param)

            # generate snippets
            snippets = []
            
            # first snippet
            fp2 = np.linspace(ppar[0], params[0], fid+2)[1:]
            snpt = np.vstack((c[0:snipi[0]+1], spl.evaluate_list(fp2.tolist())))
            snippets.append(snpt)

            # inbetween snippets
            for i in range(len(snipi)-1):
                fp1 = np.linspace(params[i], npar[i], fid+2)[0:-1]
                fp2 = np.linspace(ppar[i+1], params[i+1], fid+2)[1:]
                snpt = np.vstack((spl.evaluate_list(fp1.tolist()), c[snipi[i]+1:snipi[i+1]+1], spl.evaluate_list(fp2.tolist())))
                snippets.append(snpt)

            # final snippet
            fp1 = np.linspace(params[-1], npar[-1], fid+2)[0:-1]
            snpt = np.vstack((spl.evaluate_list(fp1.tolist()), c[snipi[-1]+1:]))
            snippets.append(snpt)

            return snippets


        @ staticmethod
        def split(c: list|tuple|np.ndarray, indx: list|tuple|np.ndarray, fid: int, repac: float = 4, tol = 0.01) -> list:
            """
            Split a curve into seperate curves at selected indexes. The curve must be well repressented for this to work properly.

            Args:
                c: curve
                indx: the indexes at which to split the curve
                fid: the greater the fid, the more accurate the segments are to the original curve
                repac: representation accuracy, the higher this is, the higher the accuracy of the function, depending on the accuracy that the curve repressents itself
                tol: geo.spline.get_prmtr tolerance

            Returns:
                list of curve segments
            
            """
            if len(np.shape(indx)) == 1:
                if len(indx) == 0:
                    return [c]
            indx = np.array(indx)
            c = np.array(c)
            spl = geo.spline.fit(c)            
            seglen = geo.curve.segment_len(c)
            clen = np.insert(np.cumsum(seglen),0,0)
            delta = 1 / (repac/min(seglen/clen[-1]) + 1)
            params = geo.spline.get_prmtr(spl, 'point', c[indx], tol, delta)
            params.sort()

            # find next and previous parameters
            ppar = geo.spline.get_prmtr(spl, 'point', c[indx-1], tol, delta)
            npar = geo.spline.get_prmtr(spl, 'point', c[indx+1], tol, delta)

            # generate splinters
            splinters = []
            
            # first splinter
            fp2 = np.linspace(ppar[0], params[0], fid+2)[1:]
            splt = np.vstack((c[0:indx[0]], spl.evaluate_list(fp2.tolist())))
            splinters.append(splt)

            # inbetween splinters
            for i in range(len(indx)-1):
                fp1 = np.linspace(params[i], npar[i], fid+2)[0:-1]
                fp2 = np.linspace(ppar[i+1], params[i+1], fid+2)[1:]
                splt = np.vstack((spl.evaluate_list(fp1.tolist()), c[indx[i]+1:indx[i+1]], spl.evaluate_list(fp2.tolist())))
                splinters.append(splt)

            # final splinter
            fp1 = np.linspace(params[-1], npar[-1], fid+2)[0:-1]
            splt = np.vstack((spl.evaluate_list(fp1.tolist()), c[indx[-1]+1:]))
            splinters.append(splt)

            return splinters


        # Interactions
        @staticmethod
        def mean(c1: list|tuple|np.ndarray, c2: list|tuple|np.ndarray, wc: float = 0.5) -> np.ndarray:
            """
            Calculate the mean curve.

            Args:
                c1: curve1
                c2: curve2
                wc: weight coeficient

            Returns:
                mean curve

            """
            c1, c2 = np.array(c1), np.array(c2)
            if len(c1) == len(c2) == 2:
                return wc*c1 + (1-wc)*c2
            # else
            spl1 = geo.spline.fit(c1)
            spl2 = geo.spline.fit(c2)
            ppar = geo.spline.get_prmtr(spl1, 'point', c1[1:-1])
            npar = geo.spline.get_prmtr(spl2, 'point', c2[1:-1])
            mpar = np.sort(np.hstack((ppar, npar)))
            crvp1 = np.vstack((c1[0], spl1.evaluate_list(mpar), c1[-1]))
            crvp2 = np.vstack((c2[0], spl2.evaluate_list(mpar), c2[-1]))
            return wc * crvp1 + (1-wc) * crvp2


        @staticmethod
        def intersect(c1: list|tuple|np.ndarray, c2: list|tuple|np.ndarray, only1: bool = False) -> list:
            """
            Find the intersections between two curves.

            Args:
                c1: curve 1
                c2: curve 2
                only1: if True, returns only the first intersection and stops, helpfull to quicken specific cases 
            
            Returns:
                list containing lists that each contains:
                -p (ndarray): [x, y] coordinates of the intersection
                -i (int): index of coordinates of last point before the intersection, of the curve 1 
                -j (int): index of coordinates of last point before the intersection, of the curve 2 
            
            """
            c1, c2 = np.array(c1), np.array(c2)
            inters = []
            stop = False
            for i in range(0, np.shape(c1)[0]-1):
                lf1 = geo.line.fit(c1[i], c1[i+1])
                # get segments crossing the line
                ydiff = c2[:,1] - np.polyval(lf1, c2[:,0])
                intersj = np.nonzero(ydiff[0:-1]*ydiff[1:] < 0)[0]
                for j in intersj:
                    # Find intersection
                    lf2 = geo.line.fit(c2[j], c2[j+1])
                    p0 = geo.line.intersect(lf1, lf2)

                    # Check if p0 is internal in segment of curve 1
                    pintbool = min([c1[i,0], c1[i+1,0]]) <  p0[0] < max([c1[i,0], c1[i+1,0]]) 

                    if pintbool:
                        inters.append([p0,i,j])
                        if only1:
                            stop = True
                            break
                if stop:
                    break

            return inters


        @staticmethod
        def arrange(c1: list|tuple|np.ndarray, c2: list|tuple|np.ndarray, proxd: float) -> list:
            """
            Arrange two curves together so that the firsts last point is at the seconds first point by flipping theri sequencing.
            The two curves must already have a common end point. If they have multiple common endpoints so that they form a loop,
            arange them so that the first curve retains its sequencing.

            Args:
                c1: curve 1
                c2: curve 2
                proxd: distance at which points are considered to overlap
            
            Returns:
                list containing c1 and c2 properly arranged
            
            Raises:
                'No overlapping points' when no points are within proxd'

            """
            c1, c2 = np.array(c1), np.array(c2)
            if geo.inproxim(c1[-1], c2[0], proxd):
                return [c1, c2]
            elif geo.inproxim(c1[-1], c2[-1], proxd):
                return [c1, np.flipud(c2)]
            elif geo.inproxim(c1[0], c2[0], proxd):
                return [np.flipud(c1), c2]
            elif geo.inproxim(c1[0], c2[-1], proxd):
                return [np.flipud(c1), np.flipud(c2)]
            else:
                raise Exception('curve.arrange Exception: No overlapping points')


        @staticmethod
        def patch(clist: list) -> np.ndarray:
            """
            Patch curves into one (the curves must be arranged properly).

            Args:
                c1: curve 1
                c2: curve 2

            Returns:
                singular curve

            """
            for i in range(len(clist)-1):
                clist[i] = clist[i][0:-1]
            return np.vstack(clist)


        @staticmethod
        def bridge(c1: list|tuple|np.ndarray, c2: list|tuple|np.ndarray) -> np.ndarray:
            """
            Compute a curve to connect two disconnected curves (that dont have a common point) at the end of the first and the start of the second.

            Args:
                c1: curve 1
                c2: curve 2
            
            Returns:
                bridging curve
            
            """
            c1, c2 = np.array(c1), np.array(c2)
            p0 = geo.line.intersect(geo.line.fit(c1[-1], c1[-2]), geo.line.fit(c2[0], c2[1]))
            bez = geo.spline.bezier([c1[-1], p0, c2[0]])
            bez.delta = 10**-2
            bezlen = geo.curve.length(bez.evalpts)[-1]
            minseg = min(np.hstack((geo.curve.segment_len(c1), geo.curve.segment_len(c2))))
            bez.delta = 1 / (bezlen/minseg + 2)
            return np.array(bez.evalpts)


        @staticmethod
        def fillet_bez(c1: list|tuple|np.ndarray, c2: list|tuple|np.ndarray, fl: float) -> np.ndarray:
            """
            Fillet two patchable curves with a bezier curve.

            Args:
                c1: curve 1
                c2: curve 2
                fl: fillet length

            Returns:
                singular curve
            
            """
            c1, c2 = np.array(c1), np.array(c2)

            # check for intersections
            c2 = np.flipud(c2)
            intersection = geo.curve.intersect(c1[1:-1], c2[1:-1], True)
            if len(intersection) > 0:
                b, p, i, j = geo.curve.intersect(c1[1:-1], c2[1:-1], True)[0]
                if b:
                    c1 = np.vstack((c1[0:i+2], p))
                    c2 = np.vstack((c2[0:j+2], p))

            c1 = np.flipud(c1)
            c2 = np.flipud(c2)

            # find tangent points
            p1 = geo.curve.interp(c1, 'length', fl)[0]
            p2 = geo.curve.interp(c2, 'length', fl)[0]
            # shorten curves
            c1 = c1[np.nonzero(geo.curve.length(c1) > fl)[0]]
            c2 = c2[np.nonzero(geo.curve.length(c2) > fl)[0]]
            p0 = geo.line.intersect(geo.line.fit(c1[0],p1), geo.line.fit(c2[0],p2))
            bcrv = geo.spline.bezier([p1,p0,p2])
            # calculate density
            bcrv.delta = 0.05
            dens = 1 / min(np.hstack((geo.curve.segment_len(c1), geo.curve.segment_len(c2))))
            n = int(1.5*(np.pi - abs(geo.vector.angle(p1-p0, p2-p0))) + dens * geo.curve.length(bcrv.evalpts)[-1]) + 2
            # get points
            bcrv.delta = 1/(1+n)
            return np.vstack((np.flipud(c1), bcrv.evalpts, c2))


        @staticmethod
        def fillet_arc(c1: list|tuple|np.ndarray, c2: list|tuple|np.ndarray, r: float) -> np.ndarray:
            """
            Fillet two patchable curves with a circular arc.

            Args:
                c1: curve 1
                c2: curve 2
                r: radius of fillet arc

            Returns:
                singular curve
            
            """
            c1, c2 = np.array(c1), np.array(c2)

            # Check for intersections
            c2 = np.flipud(c2)
            intersection = geo.curve.intersect(c1[1:-1], c2[1:-1], True)
            if len(intersection) > 0:
                b, p, i, j = geo.curve.intersect(c1[1:-1], c2[1:-1], True)[0]
                if b:
                    c1 = np.vstack((c1[0:i+2], p))
                    c2 = np.vstack((c2[0:j+2], p))

            c1 = np.flipud(c1)
            c2 = np.flipud(c2)

            # To reduce hopeless looping which eats up resources, if finding a
            # tangent circle of any radius fails too many consecutive times on 
            # the same segment (failim times) the function skips the segment
            # and continues on the next one.
            ifailim = 5
            jfailim = 10

            # Find tangent circle
            failed_tangent = True
            ifails = 0
            rdiff = np.inf
            for i in range(0, np.shape(c1)[0]-1):
                jfails = 0
                jsuccess = False
                for j in range(0, np.shape(c2)[0]-1):

                    rlims = geo.circle.tang2seg(c1[i:i+2], c2[j:j+2])

                    # Check consecutive failures
                    if len(rlims) == 0:
                        if jsuccess:
                            jfails += 1
                            if jfails > jfailim:
                                break
                        continue
                    else:
                        jsuccess = True
                        jfails = 0
                    
                    # Check if a segment combination that satisfies r perfectly is found
                    if rlims[0] < r < rlims[1]:
                        failed_tangent = False
                        break
                    else:
                        # Select the best suited r and add it to the pool
                        rbest = rlims[np.argmin(np.abs(rlims - r))]

                        if np.abs(rbest - r) < rdiff:
                            rdiff = np.abs(rbest - r)
                            rcl = rbest
                            icl = i
                            jcl = j
                        else:
                            jfails += 1

                if not jsuccess:
                    ifails += 1
                else:
                    ifails = 0
                
                if (not failed_tangent) or (ifails > ifailim):
                    break

            if failed_tangent:      # If perfect fillet fails, get aproximate fillet
                rstr = str(r)
                r = rcl
                i = icl
                j = jcl
                print('gep.curve.fillet_arc Warning: couldnt do requested fillet of radius ' + rstr + ', doing a fillet of radius ' + str(r) + ' instead.')
            
            # Get tangent circle
            quadrnt = geo.vector.quadrant(c1[i+1] - c1[i], c2[j+1] - c2[j])
            lf1 = geo.line.fit(c1[i], c1[i+1])
            lf2 = geo.line.fit(c2[j], c2[j+1])
            p0, ptan1, ptan2 = geo.circle.tang2lnr(lf1, lf2, r, quadrnt)

            # Delete points before tangency
            c1 = c1[i+1:]
            c2 = c2[j+1:]

            # Generate fillet arc
            # Number of points
            ang = abs(geo.vector.angle(ptan1 - p0, ptan2 - p0))
            dens = 1 / min(np.hstack((geo.curve.segment_len(c1), geo.curve.segment_len(c2))))
            n = int(ang * (1.5 + dens * r)) + 2
            
            # Generate arc
            ca = geo.circle.arc(ptan1, ptan2, p0, n)

            # Assemble curve
            c1 = np.flipud(c1)
            return np.vstack((c1, ca, c2))


    # CURVE FORGE
    class crvforge:
        """
        Functions for generating curves based on given curves. Very usefull for generating the curves of high lift elements.
        """
        @staticmethod
        def varioff(c: list|tuple|np.ndarray, offset: Callable[[np.ndarray], np.ndarray]|list|tuple|np.ndarray) -> np.ndarray:
            """
            Varriable offset. Moves a curves points upon its normals.

            Args:
                c: curve
                offset: can be either a function that is used to calculate the normal displacement of each point, that decides values lengthwise, and should take values from 0 (beginning of curve) to 1 (end of curve)
                    or an array-like with the displacement values for each curve point. 
            
            Returns:
                generated curve        
            
            """
            if not (type(offset) == tuple or type(offset) == list or type(offset) == np.ndarray):
                offset = offset(geo.curve.length(c, True))
            offset = np.transpose([offset, offset])
            return c + geo.curve.normals(c) * offset


        @staticmethod
        def bezctrl(c: list|tuple|np.ndarray, m: int|str, gens: int, w: list|tuple|np.ndarray = 1) -> np.ndarray:
            """
            Generate a bezier curve using the given curves points as control points

            Args:
                c: curve
                m:  2 <= m <= np.shape(c)[0], number of points of the curve to take as control points, if m = 'all', take all points
                gens: number of generations (each generation is based on the previous curve)
                w: weights of the control points, length must be equal to the (int) m argument
            
            Returns:
                generated curve

            """
            for i in range(gens):
                if not (type(m) == str):
                    c = c[np.array(np.round(np.linspace(0, np.shape(c)[0]-1, m)), dtype=int)]
                spl = geo.spline.bezier(c, w)
                spl.delta = 1/(len(c)+1)
                c = np.array(spl.evalpts)

            return c


        @staticmethod
        def arcross(c: list|tuple|np.ndarray, p: list|tuple|np.ndarray = []) -> np.ndarray:
            """
            Generate an arc curve, crossing from the first to the last point of the curve.

            Args:
                c: curve
                p: if given the arc will be drawn so the circle it generates fits this point, if not given the arc will be tangent to the first curve segment instead.
            
            Returns:
                generated curve

            """
            if len(p) == 0:
                p0 = geo.circle.tang2pln(geo.line.fit(c[0], c[1]), c[0], c[-1])
            else:
                p0 = geo.circle.fit([p, c[0], c[-1]])[0]
            return geo.circle.arc(c[0], c[-1], p0, len(c))


        @staticmethod
        def weav(c: list|tuple|np.ndarray, weight: Callable[[np.ndarray], np.ndarray]|list|tuple|np.ndarray, p: list|tuple|np.ndarray = []) -> np.ndarray:
            """
            Generate a curve by calculating the weighted average of its points and the given(s)

            Args:
                c: curve
                p: point, if this is a list of many points instead of a single, it should have the same size as the curve
                weight: can be either a function that is used to calculate the weight of each point that decides values lengthwise, and should take values from 0 (beginning of curve) to 1 (end of curve)
                    or an array-like with the displacement values for each curve point. 

            Returns:
                generated curve

            """
            if len(p) == 0:
                p = (c[0] + c[1]) / 2
            if len(np.shape(p)) == 1:
                p = np.tile(p, (len(c),1))
            if not (type(weight) == tuple or type(weight) == list or type(weight) == np.ndarray):
                weight = weight(geo.curve.length(c, True))
            weight = np.transpose([weight, weight])
            return weight * c + (1-weight) * p


        # THIS ONE'S SPECIAL
        @staticmethod
        def shortcut(c: list|tuple|np.ndarray, w: list|tuple|np.ndarray) -> np.ndarray:
            """
            Generate a curve connecting the two ends of the given curve, by calculating average points between its sides. While convoluted,
            its the best function for generating slot curves on the trailing edge. Its drawbacks are that it cant be passed directly to the
            Airfoil.slot() function (the curve it generates should be passed instead), and only works on airfoily-looking curves.

            Args:
                c: curve
                w: weights, contains pairs of two values, normalized altitudes and weights, all normalized altitudes take values from 0 to 1

            Returns:
                generated curve
            
            """
            c, w = np.array(c), np.array(w)
            e1 = (c[0] + c[-1]) / 2
            e2 = c[np.argmax(geo.distance([c[0]], c)[0] + geo.distance([c[-1]], c)[0])]
            theta = geo.vector.angle(e1-e2)
            c = geo.rotate(c, e2, -theta-np.pi/2)
            ymin = max(c[0,1], c[-1,1])
            ymax = e2[1]
            yvals =  ymin + w[:,0] * (ymax - ymin)
            lf = np.transpose([np.zeros(len(yvals)), yvals])
            ppairs = geo.curve.interp(c, 'linear', lf)
            w[:,0] = w[:,1]
            c = ppairs[0:-1:2] * (1 - w) + ppairs[1::2] * w
            return geo.rotate(c, e2, theta+np.pi/2)


    # Shape Object Class
    class Shape:
        """
        A class repressenting a geometric shape, as a collection of curves. Idealy curves should have common end coordinates so as to, when attached, they create a surface.
        """
        def __init__(self, curves: list|tuple):
            self.curves = list(curves)
            for i in range(len(self.curves)):
                self.curves[i] = np.array(self.curves[i])
            self.transformlog = []


        def points(self) -> np.ndarray:
            """
            Get all the points compromising the shape.

            Returns:
                N x 2 array with point coordinates

            """
            return np.vstack(self.curves)

        
        def translate(self, tv: list|tuple|np.ndarray):
            """
            Translate the shape by vector tv.

            Args:
               tv: [x, y] components of displacement vector 

            """
            for i in range(len(self.curves)):
                self.curves[i] = geo.translate(self.curves[i], tv)
            self.transformlog.append(['translate', tv])
        

        def rotate(self, center: list|tuple|np.ndarray, theta: float):
            """
            Rotate shape around center by theta.

            Args:
                center: [x, y] coordinates of rotation center
                theta: the angle of rotation in radiants

            """
            for i in range(len(self.curves)):
                self.curves[i] = geo.rotate(self.curves[i], center, theta)
            self.transformlog.append(['rotate', center, theta])
        

        def scale(self, center: list|tuple|np.ndarray, fv: list|tuple|np.ndarray):
            """
            Scale shape around center accodring to vector fv.

            Args:
                center: [x, y] coordinates of rotation center
                fv: [xf, yf] factors by which each coordinate is scaled

            """
            for i in range(len(self.curves)):
                self.curves[i] = geo.scale(self.curves[i], center, fv)
            self.transformlog.append(['scale', center, fv])
            

        def mirror(self, ax: list|tuple|np.ndarray):
            """
            Mirror shape around axis ax.

            Args:
                ax: [[xa1, ya1], [xa2, ya2]] a matrix of two points that define the mirroring axis 
            
            """
            for i in range(len(self.curves)):
                self.curves[i] = geo.mirror(self.curves[i], ax)
            self.transformlog.append(['mirror', ax])


        def transform(self, tv: list|tuple|np.ndarray = [0,0],
                      rc: list|tuple|np.ndarray = [0,0], theta: float = 0,
                      sc: list|tuple|np.ndarray = [0,0], fv: list|tuple|np.ndarray = [1,1],
                      mb: bool = False, ax: list|tuple|np.ndarray = [[0,0], [0,0]]):
            """
            Transform the shape according to arguments. The rotation and scaling centers, along with the mirror axis
            are considered relative to the shape, and they too will be subject to the transformation as it happens.

            Args:
                tv: [x, y] components of displacement vector
                rc: [x, y] coordinates of rotation center
                theta: the angle of rotation in radiants
                sc: [x, y] coordinates of rotation center
                fv: [xf, yf] factors by which each coordinate is scaled
                mb: if True, mirror, else dont
                ax: [[xa1, ya1], [xa2, ya2]] a matrix of two points that define the mirroring axis 
            
            """
            self.translate(tv)
            rc = geo.translate(rc, tv)
            sc = geo.translate(sc, tv)
            ax = geo.translate(ax, tv)
            self.rotate(rc, theta)
            sc = geo.rotate(sc, rc, theta)
            ax = geo.rotate(ax, rc, theta)
            self.scale(sc, fv)
            ax = geo.scale(ax, fv)
            if mb:
                self.mirror(ax)


        def undo_transform(self, n: int = None):
            """
            Undo a certain number of transformations.

            Args:
                n: the number of transformations to undo, if None undo all transformations

            """
            if n == None:
                n = len(self.transformlog)
            for i in range(n):
                tsf = self.transformlog[-1]
                if tsf[0] == 'translate':
                    self.translate([-tsf[1][0], -tsf[1][1]])
                elif tsf[0] == 'rotate':
                    self.rotate(tsf[1], -tsf[2])
                elif tsf[0] == 'scale':
                    self.scale(tsf[1], [1/tsf[2][0], 1/tsf[2][1]])
                elif tsf[0] == 'mirror':
                    self.mirror(tsf[1])
                self.transformlog.pop(-1), self.transformlog.pop(-1)


        def plot(self, show: bool = False, **plotargs):
            """
            Plot the shape.

            Args:
                plotargs: arguments to pass to the pyplot.plot function
            
            """
            for curve in self.curves:
                geo.curve.plot(curve, False, **plotargs)
            if show:
                pyplot.grid()
                pyplot.axis('equal')
                pyplot.show()


        def sort(self, proxd: float, start: int = 0):
            """
            Sort indexes of the curves in the shape list, and reverse curve where needed,
            so that they are aranged the same way they meet each other in geometric space.

            Args:
                proxd: proximity distance for two curve end points to be considered common
                start: the index of the curve in the shape that will be considered first

            """
            sc = [self.curves[start]]
            oc = list(self.curves)
            oc.pop(start)
            for _i in range(len(oc)):
                keyp = sc[-1][-1]
                for curve in oc:
                    if geo.inproxim(keyp, curve[0], proxd):
                        sc.append(curve)
                        oc.remove(curve)
                    elif geo.inproxim(keyp, curve[-1], proxd):
                        sc.append(np.flipud(curve))
                        oc.remove(curve)
            self.curves = sc


        def isclockwise(self) -> bool:
            """
            Determine if the shape's order follows a clockwise manner.
            
            Returns:
                True if the shape follows clockwise order

            """
            p = self.points()
            rl = 1.1 * max(geo.distance([[0,0]],p)[0])
            v = geo.vector.unit((p[0] + p[1])/2)
            ray = [[0,0], rl * v]
            # trace ray on shape and get points
            inters = geo.curve.intersect(ray, p)
            p0s = np.zeros((len(inters), 2))
            for i in range(len(inters)):
                p0s[i] = inters[i][0]
            # get furthest away trace
            imax = np.argmax(geo.distance([[0,0]], p0s)[0])
            pi = p[inters[imax][2]+1]
            p0 = p0s[imax]
            if geo.vector.angle(-p0, pi-p0) >= 0:
                return False
            else:
                return True
            

        def reverse(self):
            """
            Reverse the shapes order by reversing the curves and their indexing. The shape must be sorted.
            """
            for i in range(len(self.curves)):
                self.curves[i] = np.flipud(self.curves[i])
            self.curves.reverse()


        def vertex_angle(self) -> np.ndarray:
            """
            Measure the internal angle of each vertex of the shape. The shape must be sorted.

            Returns:
                array containing the internal angles
            
            """
            curves = self.curves
            lc = len(curves)
            ci = list(range(lc)) + [0]
            if self.isclockwise():
                ev = 0
            else:
                ev = -2*np.pi
            angles = np.zeros(lc)
            for i in range(lc):
                j = ci[i+1]
                v1 = curves[i][-2] - curves[i][-1]
                v2 = curves[j][1] - curves[j][0]
                vang = geo.vector.angle(v1, v2)
                if vang > 0:
                    vang = vang + ev
                elif vang < 0:
                    vang = vang + np.pi*2 + ev
                angles[i] = vang

            return angles


        def round_vertex(self, indx: list|tuple|np.ndarray, ln: float, rnd: int):
            """
            Dull the shape's vertexes by rounding them (or cutting them off). The shape must be sorted.

            Args:
                indx: contains the indexes of the first of the two meeting curves, for every vertex that will be dulled
                ln: the length that measures how far up the edge the rounding/cut will go
                rnd: roundness, the greater the number, the rounder the edge will be, at even numbers the edge is flat, and at odd numbers it is sharp (less than before)

            """
            curves = self.curves
            indx = np.array(indx)
            rnd = rnd + 2
            if rnd % 2 == 0:
                indx = indx + np.arange(len(indx))
                ci = list(range(len(curves) + len(indx) - 1)) + [0]
            else:
                ci = list(range(len(curves))) + [0]

            for i in indx:
                j = ci[i+1]
                # first curve
                c = curves[i]
                if geo.curve.length(c)[-1] < 3*ln:    # if curve too short, skip
                    print('geo.Shape.round_vertex Warning: curve too short to round at requested length, skipping vertex')
                    continue
                cm, c1e = geo.curve.snip(c, 'length', geo.curve.length(c)[-1]-ln, 3)
                buc = c
                curves[i] = cm

                # second curve
                c = curves[j]
                if geo.curve.length(c)[-1] < 3*ln:
                    print('geo.Shape.round_vertex Warning: curve too short to round at requested length, skipping vertex')
                    curves[i] = buc
                    continue
                c2e, cm = geo.curve.snip(c, 'length', ln, 3)
                curves[j] = cm

                if rnd % 2 == 0:
                    ce = geo.curve.patch([c1e, c2e])
                    bz = geo.spline.bezier(ce)
                    bz.delta = 1/rnd
                    ce = np.array(bz.evalpts)
                    ce1 = ce[0:int(rnd/2)]
                    ce2 = ce[int(rnd/2):]
                    curves[i] = geo.curve.patch([curves[i], ce1])
                    curves[j] = geo.curve.patch([ce2, curves[j]])
                    curves.insert(i+1, np.array([ce1[-1], (ce1[-1] + ce2[0]) / 2, ce2[0]]))
                else:
                    ce = geo.curve.patch([c1e, c2e])
                    bz = geo.spline.bezier(ce)
                    bz.delta = 1/rnd
                    ce = np.array(bz.evalpts)
                    ce1 = ce[0:int((rnd+1)/2)]
                    ce2 = ce[int((rnd-1)/2):]
                    curves[i] = geo.curve.patch([curves[i], ce1])
                    curves[j] = geo.curve.patch([ce2, curves[j]])


        def fillet(self, indx: list|tuple|np.ndarray, method: str, argval: float):
            """
            Fillet the shape's vertexes. The shape must be sorted.

            Args:
                indx: contains the indexes of the first of the two meeting curves, for every vertex that will be filleted
                method: the filleting method, either "bez" or "arc"
                argval: the argument value for the filleting function, look to geo.curve.fillet_bez / geo.curve.fillet_arc

            """
            curves = self.curves
            ci = list(range(len(curves)-len(indx))) + [0]
            indx = np.array(indx)
            indx = indx - np.arange(len(indx))
            if method == 'arc':
                filletfunc = geo.curve.fillet_arc
            elif method == 'bez':
                filletfunc = geo.curve.fillet_bez

            for i in indx:
                j = ci[i+1]
                c1, c2 = curves[i], curves[j]
                cf = filletfunc(c1, c2, argval)
                self.curves.pop(i)
                self.curves.insert(i, cf)
                self.curves.pop(j)
                

        def geofid(self, n: int):
            """
            Change the geometric fidelity by changing number of points of the airfoil, using a 3rd degree spline.

            Args:
                n: the new number of points

            """
            splines = []
            lens = []
            for curve in self.curves:
                lens.append(geo.curve.length(curve)[-1])
                splines.append(geo.spline.fit(curve))
            
            lens = np.array(lens)
            totalen = sum(lens)
            lf = lens / totalen
            n = np.array(lf * n, dtype=int)
            
            curves = []
            for i, spl in enumerate(splines):
                spl.delta = 1/(n[i]+2)
                curves.append(np.array(spl.evalpts))
            
            self.curves = curves


        def splitin3(self, lf: list|tuple|np.ndarray, fid: int = 0, tol: float = 10**-3) -> list:
            """
            Split a shape into three lists of curves using a line.

            Args:
                lf: line factors
                fid: fidelity of the split, see geo.curve.snip()
            
            Returns:
                list of three lists of curves
            
            """
            ic = np.arange(len(self.curves))
            
            for i in ic:
                snpt = geo.curve.snip(self.curves[i], 'linear', lf, fid, tol=tol)
                if len(snpt) > 1:
                    ic1 = i
                    snpt1 = snpt
                    break
            for i in np.flip(ic):
                snpt = geo.curve.snip(self.curves[i], 'linear', lf, fid, tol=tol)
                if len(snpt) > 1:
                    ic2 = i
                    snpt2 = snpt
                    break
            
            if ic1 == ic2:
                cl1 = self.curves[0:ic1] + [snpt1[0]]
                cl2 = [geo.curve.patch(snpt1[1:-1])]
                cl3 = [snpt1[-1]] + self.curves[ic1+1:]
            else:
                cl1 = self.curves[0:ic1] + [snpt1[0]]
                cl2 = [geo.curve.patch(snpt1[1:])] + self.curves[ic1+1:ic2] + [geo.curve.patch(snpt2[0:-1])]
                cl3 = [snpt2[-1]] + self.curves[ic2+1:]

            return [cl1, cl2, cl3]


        def airfoil(self):
            """
            Returns an Airfoil object constructed from the current shape.
            """
            afl = Airfoil(list(self.curves))
            afl.transformlog = self.transformlog
            return afl


    # Section Object Class
    class Section:
        """
        A class repressenting a section of geometric shapes.
        """
        def __init__(self, shapes: list|tuple):
            self.shapes = list(shapes)

        
        def translate(self, tv: list|tuple|np.ndarray):
            """
            Translate section by vector tv.

            Args:
               tv: [x, y] components of displacement vector 

            """
            for shape in self.shapes:
                shape.translate(tv)
        

        def rotate(self, center: list|tuple|np.ndarray, theta: float):
            """
            Rotate section around center by theta.

            Args:
                center: [x, y] coordinates of rotation center
                theta: the angle of rotation in radiants

            """
            for shape in self.shapes:
                shape.rotate(center, theta)
        

        def scale(self, center: list|tuple|np.ndarray, fv: list|tuple|np.ndarray):
            """
            Scale section around center accodring to vector fv.

            Args:
                center: [x, y] coordinates of rotation center
                fv: [xf, yf] factors by which each coordinate is scaled

            """
            for shape in self.shapes:
                shape.scale(center, fv)
        

        def mirror(self, ax: list|tuple|np.ndarray):
            """
            Mirror section around axis ax.

            Args:
                ax: [[xa1, ya1], [xa2, ya2]] a matrix of two points that define the mirroring axis 
            
            """
            for shape in self.shapes:
                shape.mirror(ax)


        def plot(self, show: bool = False, **plotargs):
            """
            Plot the section.

            Args:
                plotargs: arguments to pass to the pyplot.plot function
            
            """
            for shape in self.shapes:
                shape.plot(False, **plotargs)
            if show:
                pyplot.grid()
                pyplot.axis('equal')
                pyplot.show()


class Airfoil(geo.Shape):
    """
    Shape repressenting an airfoil. First point of first curve must be at the trailing edge.
    """
    def __init__(self, curves, te_round_if_open = False):
        super().__init__(curves)
        # sort and rearange airfoil
        self.sort(10**-6, 0)
        if self.isclockwise():
            self.reverse()

        # get trailing edge
        te1 = self.curves[0][0]
        te2 = self.curves[-1][-1]
        te = (te1 + te2)/2

        # close trailing edge if open
        if not geo.inproxim(te1, te2, 10**-6):
            if te_round_if_open:
                rp = 4
                peval = np.linspace(0,1,rp).tolist()
                bez = geo.spline.bezier([te, self.curves[0][0], self.curves[0][1]])
                self.curves[0] = geo.curve.patch([bez.evaluate_list(peval), self.curves[0][1:]])
                bez = geo.spline.bezier([self.curves[-1][-2], self.curves[-1][-1], te])
                self.curves[0] = geo.curve.patch([self.curves[-1][0:-1], bez.evaluate_list(peval)])
            else:
                self.curves.append(np.array([te2, te1]))

        self.__generate_attr__()


    def __generate_attr__(self):
        """
        Generate airfoil attributes.
        """
        # Get trailing edge
        self.te = self.curves[0][0]
        # Get leading edge
        clp = []
        for i in range(len(self.curves)):
            dist = geo.distance([self.te], self.curves[i])[0]
            j = np.argmax(dist)
            clp.append([j,dist[j]])
        clp = np.array(clp)
        i = np.argmax(clp[:,1])
        j = int(clp[i,0])
        self.le = self.curves[i][j]

        # Get AoA
        self.aoa = - geo.vector.angle(self.te - self.le)

        # Get chord
        self.chord = geo.distance([self.te],[self.le])[0][0]

        # Get leading edge circle
        if j == len(self.curves[i]) or j == 0:
            self.le_crcl = [self.le, 0]
        else:
            self.le_crcl = geo.circle.fit(self.curves[i][j-2:j+3])
        
        # Get profile
        profile = self.curves[0]
        for curve in self.curves[1:]:
            profile = np.vstack((profile, curve[1:]))
        self.profile = profile


    def interp(self, x: float) -> list:
        """
        Interpolate the airfoil curves at x.

        Args:
            x: the x  value at which to interpolate
        
        Returns:
            list containing all interpolated point coordinates
        
        """
        interpgroup = []
        for curve in self.curves:
            interpgroup += list(geo.curve.interp(curve, 'linear', [10**6, -10**6 * x]))
        return interpgroup


    def reset(self):
        """
        Set the airfoil to have it's leading edge at 0,0 and trailing edge at 100,0.
        """
        self.translate(-self.le)
        self.rotate(-self.aoa)
        self.scale(100/self.chord)
        self.__generate_attr__()


    # OVERRIDEN METHODS
    def translate(self, tv):
        super().translate(tv)
        self.__generate_attr__()
    

    def rotate(self, center, theta):
        super().rotate(center, theta)
        self.__generate_attr__()
    

    def scale(self, center, fv):
        super().scale(center, fv)
        self.__generate_attr__()
    

    def mirror(self, ax):
        super().mirror(ax)
        self.__generate_attr__()
    

    def round_vertex(self, indx, ln, cut = False):
        super().round_vertex(indx, ln, cut)
        self.__generate_attr__()
    

    def fillet(self, indx, method, argval):
        super().fillet(indx, method, argval)
        self.__generate_attr__()
    

    def geofid(self, n):
        super().geofid(n)
        self.__generate_attr__()


    # HIGH LIFT DEVICES
    def slot(self, sx: float, px: float, **crvargs) -> list:
        """
        Generate two seperate elements, that when joint together, they assume the airfoil's geometry. The elements are seperated by a curve that can either
        be explicitly passed, or generated by a crvforge function (or other similar functions), based on the airfoil profile. Keep in mind that the
        shortcut function cannot be used here effectively, instead the curve it generates should be passed explicitly.

        Args:
            sx: the suction side x, at which the elements seperate
            px: the pressure side x, at which the elements seperate
            **input_curve: explicit seperation curve
            **forge_funct: function that generates the seperation curve, if passed the keyword argument **forge_args must also be passed
            **forge_args: all the arguments that will be passed in the forge_funct, except the curve itself, if passed the keyword argument **forge_funct must also be passed

        Returns:
            list containing the two generated elements as Airfoils
        
        """
        if sx == px:
            px = sx + 10**-7
        sp = self.interp(sx)[0]
        pp = self.interp(px)[-1]
        if 'input_curve' in crvargs:
            c = crvargs['input_curve']
            c = geo.curve.fit2p(c, sp, pp, proxi_snap=True)
        else:
            ff = crvargs['forge_funct']
            fa = crvargs['forge_args']
            c = geo.curve.patch(geo.curve.snip(self.profile, 'point', [sp, pp], 0, tol=10**-5)[1:-1])
            c = ff(c, *fa)
        lf = geo.line.fit(sp, pp)
        cl1, cl2, cl3 = self.splitin3(lf, tol=10**-5)

        return [Airfoil(cl2 + [c]), Airfoil(cl1 + [c] + cl3)]


    def te_flap(self, x: float, y: float, theta: float, altsurf: bool = False):
        """
        Generate an airfoil with a trailing edge plain flap based on the original's geometry.

        Args:
            x: the x coordinate of the seperation between the main airfoil surface and the plain flap surface
            y: the y coordinate of the hinge
            theta: the deflection angle
            altsurf: if set, the x coordinate 
        
        Returns:
            Airfoil object with deflected flap
        
        """
        if theta == 0:
            return deepcopy(self)
        
        lei = np.argmax(geo.distance([self.te], self.profile))
        surfaces = [self.profile[0:lei+1], self.profile[lei:]]
        splitter = [[x-10**-7, min(self.profile[:,1])-1], [x+10**-7, max(self.profile[:,1])+1]]

        if not altsurf:
            si, opsi = 0, -1
        else:
            si, opsi = -1, 0

        p1, _, inti = geo.curve.intersect(splitter, surfaces[si])[si]
        lf = geo.line.vertical(geo.line.fit(surfaces[si][inti], surfaces[si][inti+1]), p1)
        p0 = geo.line.intersect(lf, [0, y])
        p2 = geo.curve.proxi2p(surfaces[opsi], p0)[1]
        
        cl1, cl2, cl3 = self.splitin3(geo.line.fit(p1,p2))

        le = geo.Shape(cl2)
        le.rotate(p0, theta)

        if theta > 0:
            for i in range(len(cl3)):
                for j in range(len(le.curves)-1, -1, -1):
                    inters = geo.curve.intersect(cl3[i], le.curves[j])
                    if len(inters) > 0:
                        intersdata = [i, j, inters[-1]]

            ci, cj, inters = intersdata
            pint, pi, pj = inters
            tec = [geo.curve.addpoints(cl3[ci][pi+1:], [pint], [0])] + cl3[ci+1:]
            lec = le.curves[0:cj] + [geo.curve.addpoints(le.curves[cj][0:pj+1], [pint], [1])]
            arc = geo.circle.arc(cl1[-1][-1], lec[0][0], p0, 20)
            afl = Airfoil(cl1 + [arc] + lec + tec)
        
        else:
            for i in range(len(cl1)-1, -1, -1):
                for j in range(len(le.curves)):
                    inters = geo.curve.intersect(cl1[i], le.curves[j])
                    if len(inters) > 0:
                        intersdata = [i, j, inters[-1]]

            ci, cj, inters = intersdata
            pint, pi, pj = inters
            tec = cl1[0:ci] + [geo.curve.addpoints(cl1[ci][0:pi+1], [pint], [1])]
            lec = [geo.curve.addpoints(le.curves[cj][pj+1:], [pint], [0])] + le.curves[cj+1:] 
            arc = geo.circle.arc(lec[-1][-1], cl3[0][0], p0, 20)
            afl = Airfoil(tec + lec + [arc] + cl3)

        afl.rotate(p0, -theta)
        return afl


    def le_flap(self, sx: float, px: float, theta: float):
        """
        Generate an airfoil with a leading edge flap based on the original's geometry. The results of this may be wonky, depending on the parameters.

        Args:
            sx: the x coordinate of the seperation between the main airfoil suction surface and the flap surface
            px: the x coordinate of the seperation between the main airfoil pressure surface and the flap surface (should generally be higher than sx)
            theta: the deflection angle
        
        Returns:
            Airfoil object with deflected flap
        
        """
        if sx == px:
            px = sx + 10**-7
        sp = self.interp(sx)[0]
        pp = self.interp(px)[-1]

        cl1, cl2, cl3 = self.splitin3(geo.line.fit(sp, pp))
        le = geo.Shape(cl2)
        le.rotate(cl2[-1][-1], theta)
        brdg = geo.curve.bridge(cl1[-1], le.curves[0])
        
        return Airfoil(cl1 + [brdg] + le.curves + cl3)


    def le_varcam(self, sx: float, px: float, hinge: list|tuple|np.ndarray, theta: float):
        """
        Generate an airfoil with a variable camber leading edge based on the original's geometry.

        Args:
            sx: the x coordinate of the seperation between the main airfoil suction surface and the flap surface
            px: the x coordinate of the seperation between the main airfoil pressure surface and the flap surface (should generally be higher than sx)
            hinge: the point around which the leading edge hinges, its x coordinate should always be higher than either px, sx
            theta: the deflection angle
        
        Returns:
            Airfoil object with deflected flap
        
        """
        if sx == px:
            px = sx + 10**-7
        sp = self.interp(sx)[0]
        pp = self.interp(px)[-1]
        cl1, cl2, cl3 = self.splitin3(geo.line.fit(sp, pp))

        # get point of max deflection
        maxdist = 0
        for j,c in enumerate(cl2):
            dist = geo.distance([hinge], c)[0]
            maxi = np.argmax(dist)
            if dist[maxi] > maxdist:
                maxdist = dist[maxi]
                splitdata = [j, maxi, maxdist]
        
        j, i, maxd = splitdata
        patchle = False
        if i == 0:
            scl = cl2[0:j]
            pcl = cl2[j:]
        elif i == len(cl2[j])-1:
            scl = cl2[0:j+1]
            pcl = cl2[j+1:]
        else:
            sc, pc = geo.curve.split(cl2[j], [i], 0)
            scl = cl2[0:j] + [sc]
            pcl = [pc] + cl2[j+1:]
            patchle = True

        # bend suction curves
        mind = geo.distance([hinge], [scl[0][0]])[0,0]
        for i in range(len(scl)):
            c = scl[i]
            for j in range(len(c)):
                rth = theta * (geo.distance([c[j]], [hinge])[0,0] - mind) / (maxd-mind)
                c[j] = geo.rotate(c[j], hinge, rth)
            scl[i] = c

        # bend pressure curves
        mind = geo.distance([hinge], [pcl[-1][-1]])[0,0]
        for i in range(len(pcl)):
            c = pcl[i]
            for j in range(len(c)):
                rth = theta * (geo.distance([c[j]], [hinge])[0,0] - mind) / (maxd-mind)
                c[j] = geo.rotate(c[j], hinge, rth)
            pcl[i] = c

        # patch jobs
        if patchle:
            lec = geo.curve.patch([scl[-1], pcl[0]])
            scl.pop(-1)
            pcl.pop(0)
            les = scl + [lec] + pcl
        else:
            les = scl + pcl
        les[0] = geo.curve.patch([cl1[-1], les[0]])
        les[-1] = geo.curve.patch([les[-1], cl3[0]])
        cl1.pop(-1)
        cl3.pop(0)

        return Airfoil(cl1+les+cl3)


class Mesh:
    """
    A class designed for configuring and generating mesh domains with the help of gmsh.
    """
    def __init__(self):
        self.ts_called = False
        self.ns_called = False
        self.il_called = False
        self.w_called = False
        self.cv_called = False
        self.advanced_options()


    # CONFIGURATION METHODS
    def tangent_spacing(self, default_spacing: float, smoothness_coef: float, surface_len_coef: float, proximity_coef: float, vertex_sharpness_coef: float|list, curvature_coef: float|list, orientation_coef: float|list):
        """
        Method used to pass the tangential spacing attributes of the boundary layer. The coefficients can take any
        number from 0 to inf, and the spacing is inversely proportional to them. Simply put, bigger number means more
        cells. Some coefficients can optionaly be lists with two values, each for their positive and negative counterparts.

        Args:
            default_spacing: the spacing on the boundary, unaffected by any coefficient, this is common for all curves of the shapes
            smoothness_coef: determines how the spacing will be smoothened across the curves, for a more gradual change in cell sizes, takes values between 0 and 1
            flow_len_coef: determines how much the spacing is decreased further away from the leading edge
            proximity_coef: determines how much the spacing is decreased when near other shapes
            vertex_sharpness_coef: determines how much the spacing is decreased when near a sharp vertex, optional list: different coefs for convex and reflex angles
            curvature_coef: determines how much the spacing is decreased in areas of high curvature, optional list: different coefs for positive and negative curvatures
            orientation_coef: determines how much the spacing is decreased depending on the orientation of the normal to the curve, optional list: different coefs for normal that is oriented towards positive and negative x

        """
        sp = default_spacing
        sc = smoothness_coef
        slc = surface_len_coef
        pc = proximity_coef 
        vsc = vertex_sharpness_coef 
        cc = curvature_coef 
        oc = orientation_coef

        if len(np.shape(vsc)) == 0:
            vsc = [vsc, vsc]
        if len(np.shape(cc)) == 0:
            cc = [cc, cc]
        if len(np.shape(oc)) == 0:
            oc = [oc, oc]
        
        self.bl_sp = sp
        self.bl_sc = sc
        self.bl_slc = slc
        self.bl_pc = pc
        self.bl_vsc = vsc
        self.bl_cc = cc
        self.bl_oc = oc
        self.ts_called = True


    def normal_spacing(self, first_cell_height: float, thickness: float, growth_ratio: float):
        """
        Function used to pass the normal spacing attributes of the boundary layer.
        
        Args:
            first_cell_height: the height of the first cell on the boundary
            thickness: the thickness of the boundary layer
            growth_rate: the growth_rate of the boundary layer

        """
        self.bl_fch = first_cell_height
        self.bl_t = thickness
        self.bl_gr = growth_ratio
        self.ns_called = True


    def inflation_layer(self, distance: float, near_field_spacing: float, far_field_spacing: float):
        """
        Function used to pass the inflation layer attributes.

        Args:
            distance: the distance at which the layer extends
            near_field_spacing: the spacing inside the inflation layer
            far_field_spacing: the spacing outside the inflation layer
        
        """
        self.il_d = distance
        self.il_nfs = near_field_spacing
        self.il_ffs = far_field_spacing
        self.il_called = True


    def wake_region(self, deflection_range: float, aoa: float|list, far_spacing: float, distribution_degree: int = 3):
        """
        Function used to pass the wake area attributes.

        Args:
            deflection_range: how far the flow is deflected by the airfoils
            aoa: the angle of attack for the entire domain, NOT just the geometric section, can be given as [aoa_min, aoa_max] for aoa scans
            far_spacing: the spacing of the wake area in the control volume outlet
            distribution_degree: determines for how far the wake spacing remains small (basically a polynomial degree to decide to spacing of the wake points)

        """
        self.w_dr = deflection_range
        self.w_fs = far_spacing
        self.w_dd = distribution_degree
        if len(np.shape(aoa)) == 0:
            aoa = [aoa]
        self.w_aoa = aoa
        self.w_called = True


    def control_volume(self, xrange: list, yrange: list, inlet_arc: bool = False):
        """
        Function used to pass the control volume attributes.

        Args:
            xrange: the minimum and maximum x-values of the control volume
            yrange: the minimum and maximum y-values of the control volume
            ctype: if False, control volume has a c-shape, else its rectangular

        """
        self.cv_xr = xrange
        self.cv_yr = yrange
        self.cv_ia = inlet_arc
        self.cv_called = True


    def advanced_options(self, minnodes: int = 6, thickness_var_degree: float = 0.8, lengthwise_spacing_coef_functs: list = [], fan_dens_factor: float = 0.1, wake_max_vertex_angle: float = 2*np.pi/3):
        """
        Method used to pass advanced boundary layer attributes. It is optional.

        Args:
            minnodes: the minimum number of nodes on any edge
            thickness_var_degree: how much the boundary layer thickness of an airfoil varies depending on the airfoils chord length divided by the maximum chord length of the section
            lengthwise_density_functions: list of list-pairs containing the index of the airfoil (back to front) and a lengthwise density function for it (taking values from 0 to 1) which creates an additional boundary layer density coeficient depending on the length around the airfoil
            fan_dens_factor: this multiplies the mesh density of the fans on the edges, as to avoid terrible elements
            wake_max_vertex_angle: the maximum vertex angle that produces a wake tail

        """
        self.bl_mn = minnodes
        self.bl_tvd = thickness_var_degree
        self.bl_lscf = lengthwise_spacing_coef_functs
        self.bl_fdf = fan_dens_factor
        self.w_mva = wake_max_vertex_angle


    # GENERATE MESH
    def generate(self, section: geo.Section, name: str, file_ext: str = '', review: bool = True, init_gmsh: bool = True, fin_gmsh: bool = True):
        """
        Generate a mesh for the given section according to the Mesh object's attributes using gmsh.
        Every section object must be a high fidelity airfoil. To use this function every configuration
        method must be called at least once.

        Args:
            section: the geometric section that will be meshed
            name: the name of the model
            file_ext: the mesh file extension, if this is given, a mesh file will be written as 'name' + 'file_ext'
            review: if True, open a gmsh window to review the mesh
            init_gmsh: if True, initialize gmsh
            fin_gmsh: if True, finalize gmsh

        Raises:
            Exception: All configuration methods need to be called at least once, for the generation of mesh to be possible.

        """
        if not (self.ts_called and self.ns_called and self.il_called and self.w_called and self.cv_called):
            raise Exception('All configuration methods need to be called at least once, for the generation of mesh to be possible.')
        
        # generate mesh domain data
        section = deepcopy(section)
        for i in range(len(section.shapes)):
            if type(section.shapes[i]) != Airfoil:
                section.shapes[i] = section.shapes[i].airfoil()

        print('Meshing Progress :  Aranging wing section...')
        Mesh.__arrange_section(section)
        bl_data, afl_dens = self.__boundary_layer(section)
        b_points, b_loops, b_t, b_curves, b_cf, b_nn, fp, fn = bl_data
        print('Meshing Progress :  Generating wake region data...')
        wakep = self.__wake(section, afl_dens, b_t)
        print('Meshing Progress :  Generating inflation layer data...')
        inflp = self.__inflation_layer(section, wakep)
        print('Meshing Progress :  Uncluttering domain...')
        strays = Mesh.__point_cleanup(np.vstack((wakep, inflp)))
        print('Meshing Progress :  Generating control volume data...')
        strays, inp, outp = self.__control_volume(strays)

        print('Meshing Progress :  Done generating mesh data.')
        print('Meshing Progress :  Creating CAD model...')

        # create gmsh model
        if init_gmsh:
            gmsh.initialize()
        gmsh.model.add(name)

        # add airfoil geometry
        pi = 1
        ci = 1
        li = 2
        for point in b_points:
            gmsh.model.occ.add_point(point[0], point[1], 0, 0, pi)
            pi+=1

        for curve in b_curves:
            gmsh.model.occ.add_spline(curve+1, ci)
            gmsh.model.set_entity_name(1, ci, 'airfoil')
            ci+=1

        for loop in b_loops:
            gmsh.model.occ.add_curve_loop(loop+1, li)
            li+=1

        # add control volume
        pi1 = pi
        for point in outp:
            gmsh.model.occ.add_point(point[0], point[1], 0, point[2], pi)
            pi+=1
        pi2 = pi - 1

        oci = 0
        for pio in range(pi1,pi2):
            gmsh.model.occ.add_line(pio, pio+1, ci + oci)
            oci += 1
        
        gmsh.model.occ.add_point(inp[0,0], inp[0,1], 0, inp[0,2], pi)
        gmsh.model.occ.add_point(inp[-1,0], inp[-1,1], 0, inp[-1,2], pi+1)
        gmsh.model.occ.add_line(pi1, pi, ci+oci)
        gmsh.model.occ.add_line(pi2, pi+1, ci+oci+1)

        if self.cv_ia:
            gmsh.model.occ.add_point(inp[1,0], inp[1,1], 0, inp[1,2], pi+2)
            gmsh.model.occ.add_circle_arc(pi, pi+2, pi+1, oci+ci+2, False)
            pi += 3
        else:
            gmsh.model.occ.add_line(pi,pi+1, oci+ci+2)
            pi += 2

        for oi in range(ci, ci+oci):
            gmsh.model.set_entity_name(1,oi,'outlet')
        gmsh.model.set_entity_name(1,ci+oci,'lower boundary')
        gmsh.model.set_entity_name(1,ci+oci+1,'upper boundary')
        gmsh.model.set_entity_name(1,ci+oci+2,'inlet')
        gmsh.model.occ.add_curve_loop(list(range(ci, ci+oci)) + [ci+oci, ci+oci+1, ci+oci+2], 1)
        gmsh.model.occ.add_plane_surface(np.arange(1,li), 1)
        gmsh.model.set_entity_name(2, 1, 'fluid_domain')
        
        # add stray points
        si = pi
        for point in strays:
            gmsh.model.occ.add_point(point[0], point[1], 0, point[2], si)
            si+=1
        
        gmsh.model.occ.synchronize()
        print('Meshing Progress :  Done creating CAD model.')
        print('Meshing Progress :  Setting mesh parameters...')

        # embed stray points
        gmsh.model.mesh.embed(0, np.arange(pi, si), 2, 1)

        # set point invisibility
        dimtags = np.transpose([np.zeros(si), np.arange(si)])
        gmsh.model.set_visibility(dimtags, 0)

        # add physical groups
        gmsh.model.add_physical_group(1, np.arange(1,ci), 1, 'solid')
        gmsh.model.add_physical_group(1, np.arange(ci,oci+ci+3), 2, 'fluid')
        gmsh.model.add_physical_group(2, [1], 3, 'fluid')

        # set the boundary density
        for i in range(1,ci):
            gmsh.model.mesh.set_transfinite_curve(i, b_nn[i-1], coef=b_cf[i-1])
        
        # create boundary layer fields
        for i in range(len(b_loops)):
            gmsh.model.mesh.field.add('BoundaryLayer', i+1)
            gmsh.model.mesh.field.setNumbers(i+1, 'CurvesList', b_loops[i]+1)
            gmsh.model.mesh.field.setNumber(i+1, 'Size', self.bl_fch)
            gmsh.model.mesh.field.setNumber(i+1, 'Ratio', self.bl_gr)
            gmsh.model.mesh.field.setNumber(i+1, 'Quads', 1)
            gmsh.model.mesh.field.setNumber(i+1, 'Thickness', b_t[i])
            gmsh.model.mesh.field.setNumbers(i+1, 'FanPointsList', np.array(fp)+1)
            gmsh.model.mesh.field.setNumbers(i+1, 'FanPointsSizesList', fn)

            gmsh.model.mesh.field.setAsBoundaryLayer(i+1)
        
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)

        # generate mesh
        print('Meshing Progress :  Generating mesh...')
        gmsh.model.mesh.generate(2)

        if file_ext != '':
            gmsh.write(name + file_ext)
        if review:
            gmsh.fltk.run()
        if fin_gmsh:
            gmsh.finalize()


    # HELPER FUNCTIONS
    @staticmethod
    def __get_profiles(section: geo.Section) -> list:
        """
        Get profiles of the section.
        """
        profiles = []
        for afl in section.shapes:
            profiles.append(afl.profile)
        return profiles


    @staticmethod
    def __arrange_section(section: geo.Section) -> geo.Section:
        """
        Arrange a section of airfoils so that the trailing elements are first.
        """
        unarranged = list(section.shapes)
        arranged = []
        tel = []
        lel = []
        for afl in unarranged:
            tel.append(afl.te)
            lel.append(afl.le)
        
        # find the first
        dists = geo.distance(tel,lel)
        i = np.argmax(np.max(dists, axis=1))
        arranged.append(unarranged[i])
        le = lel[i]
        unarranged.pop(i)
        lel.pop(i)
        tel.pop(i)

        # find the rest
        for _ in range(len(section.shapes)-1):
            dists = geo.distance([le], tel)[0]
            i = np.argmin(dists)
            arranged.append(unarranged[i])
            le = lel[i]
            unarranged.pop(i)
            lel.pop(i)
            tel.pop(i)

        section.shapes = arranged


    @staticmethod
    def __point_cleanup(p: np.ndarray) -> np.ndarray:
        """
        Clean up points that are uncomfortably close to each other, uncluttering the mess.
        """
        df = 1.1
        p = p[np.flip(np.argsort(p[:,2]))]
        dists = geo.distance(p, p) + np.diagflat(np.full(len(p), np.inf))
        i = 0
        while i < len(p):
            if np.min(dists[i]) < df * p[i,2]:
                p = np.delete(p, i, axis=0)
                dists = np.delete(dists, i, axis=0)
                dists = np.delete(dists, i, axis=1)
            else:
                i += 1

        return p


    @staticmethod
    def __fan_cleanup(fp: list, fn: list) -> list:
        """
        Clear the unneeded, unwanted, pariah fan points.
        """
        i = 0
        while i < len(fn):
            if fn[i] < 3:
                fn.pop(i)
                fp.pop(i)
            else:
                i+=1
        return [fp, fn]


    def __progress(self, fcw: float, lcw: float, l: float) -> list:
        """
        Decide the optimal number of points of a progression to achieve cell widths as close as possible to the ones given.
        """
        nrange = np.sort([l/fcw, l/lcw])
        nrange = [int(np.floor(nrange[0])), int(np.ceil(nrange[1]))]
        minn = self.bl_mn
        if nrange[0] == 0:
            return [minn, (lcw/fcw)**(1/minn)]
        wf = lcw/fcw
        funct = lambda n: l / sum(wf**(np.arange(int(n))/int(n))) - fcw
        n = num.chord_solver(funct, nrange, 1.1)
        n = np.ceil(n)
        if n < minn:
            n = minn
        gr = wf**(1/n)
        return [int(n), gr]


    def __split_density(self, c: np.ndarray, d: np.ndarray) -> list:
        """
        Split a geometric curve according to density fluctuation, so it can be modeled.
        """
        if len(c) > 3:
            # find points at which to split
            clen = geo.curve.length(c)
            cm1 = (clen[1:]+clen[0:-1])/2
            cm2 = (cm1[1:]+cm1[0:-1])/2
            der1 = num.derivative(clen, d)
            der1 = num.roll_average(der1, int(np.ceil(0.025*len(der1))))
            der2 = num.derivative(cm1, der1)
            der2 = num.roll_average(der2, int(np.ceil(0.025*len(der2))))
            splitl = np.sort(np.hstack((num.roots(cm1, der1, 0.1), num.roots(cm2, der2, 0.1))))

            # make sure no snippets will be too tiny for the required mesh density
            if len(splitl) > 0:
                interpi = np.searchsorted(clen, splitl)
                index = interpi + np.arange(len(interpi))
                cleninterp = np.insert(clen, interpi, splitl)
                dinterp = np.insert(d, interpi, np.interp(splitl, clen, d))
                index = np.hstack((0, index, len(cleninterp)))
                cellnum = []
                edges = np.hstack((0, splitl, clen[-1]))

                # find number of cells per snippet
                for i in range(len(index)-1):
                    cellnum.append(np.trapezoid(dinterp[index[i]:index[i+1]+1], cleninterp[index[i]:index[i+1]+1]))
                cellnum = np.array(cellnum)

                # clear all too tiny snippets
                while np.any(cellnum < self.bl_mn + 1) and len(splitl) > 1:
                    mini = np.argmin(cellnum)
            
                    if mini == 0:                            # if its on first edge merge with next
                        deli = mini
                    elif mini == len(cellnum)-1:             # if its on last edge merge with prev
                        deli = mini - 1
                    else:
                        centvals = [(edges[mini-1] + edges[mini])/2, (edges[mini] + edges[mini+1])/2, (edges[mini+1] + edges[mini+2])/2]
                        sprev, smid, snext = np.sign(np.interp(centvals, cm1, der1))
                        # select where to merge (this decides best merge based on convoluted thinking)
                        if sprev == snext:
                            if snext != smid:
                                if smid > 0:
                                    deli = mini - 1
                                else:
                                    deli = mini
                            else:
                                deli = mini - 1 + np.argmin([cellnum[mini-1], cellnum[mini+1]])
                        elif sprev == smid:
                            deli = mini - 1
                        elif snext == smid:
                            deli = mini
                        
                    # merge
                    splitl = np.delete(splitl, deli)
                    edges = np.delete(edges, deli+1)
                    cellnum[deli+1] = cellnum[deli] + cellnum[deli+1]
                    cellnum = np.delete(cellnum, deli)

            # find if to flip or not
            centval = np.hstack((0, splitl, clen[-1]))
            centval = (centval[1:] + centval[0:-1])/2
            fbl = np.interp(centval, cm2, der2) < 0
            # find first and last cell widths
            dinterp = np.interp(splitl, clen, d)
            fcwl = 1 / np.hstack((d[0], dinterp))
            lcwl = 1 / np.hstack((dinterp, d[-1]))
            # get snippets of split curve
            snippets = geo.curve.snip(c, 'length', splitl, 2)
        else:
            snippets = [c]
            fcwl, lcwl = [1/d[0]], [1/d[-1]]
            if fcwl > lcwl:
                fbl = [False]
            else:
                fbl = [True]

        # package curves
        retlist = []
        for i in range(len(fbl)):
            fcw, lcw = fcwl[i], lcwl[i]
            crv = snippets[i]
            l = geo.curve.length(crv)[-1]
            if fbl[i]:
                fcw, lcw = lcw, fcw
            n, cf = self.__progress(fcw, lcw, l)
            retlist.append([crv ,fbl[i],n,cf])

        return retlist


    def __control_volume(self, p: np.ndarray) -> list:
        """
        Generate control volume.
        """
        xmin, xmax = self.cv_xr
        ymin, ymax = self.cv_yr
        ffs = self.il_ffs
        outpi = np.nonzero(p[:,0] > xmax - (xmax - xmin) * 10**-4)[0]
        outp = p[outpi]
        p = np.delete(p, outpi, axis=0)
        outp = np.append(outp, [[xmax, ymin, ffs], [xmax, ymax, ffs]], axis=0)
        outp = outp[np.argsort(outp[:,1])]
        if self.cv_ia:
            iar = (ymax - ymin)/2
            inp = np.array([[xmin+iar, ymin, ffs], [xmin, (ymin + ymax)/2, ffs], [xmin+iar, ymax, ffs]])
        else:
            inp = np.array([[xmin, ymin, ffs], [xmin, ymax, ffs]])
        
        return [p, inp, outp]


    def __boundary_layer(self, section: geo.Section) -> list:
        """
        Generate the boundary repressentation of the section along with data to mesh its boundary layer.
        A large series of steps to ensure the best boundary layer quality that can be provided.
        """
        profiles = Mesh.__get_profiles(section)
        # find maximum chord
        maxchord = 0
        for afl in section.shapes:
            if afl.chord > maxchord:
                maxchord = afl.chord

        # generate boundary layer data for each airfoil
        bl_data = []
        afl_dens = []
        for i in tqdm(range(len(section.shapes)), desc='Meshing Progress :  Generating boundary layer data...', ascii=False, ncols=110):
            
            afl = section.shapes[i]
            # get profiles
            native_prof = profiles[i]
            foreign_prof = list(profiles)
            foreign_prof.pop(i)
            if len(foreign_prof) > 0:
                foreign_points = np.vstack(foreign_prof)
            else:
                foreign_points = []

            # calculate density for each curve
            crv_dens = []
            vertexang = geo.Shape(afl.curves).vertex_angle()
            dvang = np.insert(vertexang, 0, vertexang[-1])
            for j,crv in enumerate(afl.curves):
                crv_dens.append(self.__curve_density(crv, dvang[j:j+2], foreign_points))
            
            # create density profile
            prof_dens = crv_dens[0]
            for j in range(1,len(crv_dens)):
                if crv_dens[j][0] > prof_dens[-1]:
                    prof_dens[-1] = crv_dens[j][0]
                prof_dens = np.hstack((prof_dens, crv_dens[j][1:]))

            # apply flow len coef
            # get profile lei
            plei = np.argmax(geo.distance([native_prof[0]], native_prof)[0])
            fldens1 = np.flip(geo.curve.length(native_prof[plei::-1]))
            fldens2 = geo.curve.length(native_prof[plei+1:])
            fldens = np.hstack((fldens1, fldens2))*self.bl_slc
            prof_dens += fldens/self.bl_sp

            # apply lengthwise coef
            for lscf in self.bl_lscf:
                if lscf[0] == i:
                    prof_dens += lscf[1](geo.curve.length(native_prof, True))/self.bl_sp

            # smoothen density profile
            prof_dens = self.__smoothen_density(native_prof, prof_dens)

            # roll density
            pfl = len(prof_dens)
            prof_dens = np.hstack((prof_dens, prof_dens))
            prof_dens = num.roll_average(prof_dens, 2)
            prof_dens = np.hstack((prof_dens[pfl:int(pfl/2)+pfl],prof_dens[int(pfl/2):pfl]))

            # redistribute densities
            ei = 0
            for j in range(len(afl.curves)-1):
                crv_dens[j] = prof_dens[ei:len(afl.curves[j])+ei]
                ei += len(afl.curves[j])-1

            # save curve densities
            afl_dens.append(crv_dens)

            # gather mesh curve data for each curve
            crv_data = []
            for j, c in enumerate(afl.curves):
                crv_data.append(self.__split_density(c, crv_dens[j]))

            # calculate boundary layer thickness
            blt = self.bl_t * (afl.chord/maxchord)**self.bl_tvd 
            # create boundary repressentations and get fan points
            b_points = np.array([], dtype=float).reshape(0,2)
            b_curves, b_cf, b_nn, fp, fn = [], [], [], [], []

            # loop of loops
            ei = 0
            for j in range(len(crv_data)-1):
                fp.append(ei)
                fn.append(self.__fan_points(dvang[j], crv_dens[j][0], blt))
                for snippet_data in crv_data[j]:
                    p, fb, n, cf = snippet_data
                    b_cf.append(cf)
                    b_nn.append(n)
                    b_points = np.append(b_points, p[0:-1], axis=0)
                    cl = len(p)-1
                    if fb:
                        b_curves.append(list(range(ei+cl,ei-1,-1)))
                    else:
                        b_curves.append(list(range(ei,ei+cl+1)))
                    ei += cl  
            # last loop
            fp.append(ei)
            fn.append(self.__fan_points(dvang[-2], crv_dens[-1][0], blt))
            for snippet_data in crv_data[-1][0:-1]:
                p, fb, n, cf = snippet_data
                b_cf.append(cf)
                b_nn.append(n)
                b_points = np.append(b_points, p[0:-1], axis=0)
                cl = len(p)-1
                if fb:
                    b_curves.append(list(range(ei+cl,ei-1,-1)))
                else:
                    b_curves.append(list(range(ei,ei+cl+1)))
                ei += cl
            # last iteration
            p, fb, n, cf = crv_data[-1][-1]
            b_cf.append(cf)
            b_nn.append(n)
            b_points = np.append(b_points, p[0:-1], axis=0)
            cl = len(p)
            if fb:
                b_curves.append([0] + list(range(ei+cl-2,ei-1,-1)))
            else:
                b_curves.append(list(range(ei,ei+cl-1)) + [0])

            bl_data.append([b_points, b_curves, b_cf, b_nn, fp, fn, blt])

        # unify boundary layer data
        b_points = np.array([], dtype=float).reshape(0,2)
        b_loops, b_t, b_curves, b_cf, b_nn, fp, fn = [], [], [], [], [], [], []
        ei = 0
        eli = 0
        for bld in bl_data:
            b_points = np.append(b_points, bld[0], axis=0)
            for crv in bld[1]:
                b_curves.append(np.array(crv) + ei)
            b_loops.append(np.arange(eli, eli + len(bld[1])))
            b_t.append(bld[6])
            b_cf += bld[2]
            b_nn += bld[3]
            fp += (np.array(bld[4]) + ei).tolist()
            fn += bld[5]
            ei += len(bld[0])
            eli += len(bld[1])
        
        fp, fn = Mesh.__fan_cleanup(fp, fn)

        return [[b_points, b_loops, b_t, b_curves, b_cf, b_nn, fp, fn], afl_dens]


    def __smoothen_density(self, p: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Smoothen boundary tangential density transitions.
        """
        seglen = geo.curve.segment_len(p)
        maxdf = 1 + seglen*(1-self.bl_sc)
        sub1i = np.nonzero(maxdf < 1)[0]
        maxdf[sub1i] = 1 / maxdf[sub1i]
        df = d[1:]/d[0:-1]
        while np.any(df > maxdf) or np.any(df < 1/maxdf):
            pdti = np.nonzero(df > maxdf)[0]
            ndti = np.nonzero(df < 1/maxdf)[0]
            d[pdti] = (1+10**-4) * d[pdti + 1] / maxdf[pdti]
            d[ndti + 1] = (1+10**-4) * d[ndti] / maxdf[ndti]
            df = d[1:]/d[0:-1]
            md = max(d[-1], d[0])
            d[-1], d[0] = md, md
        
        return d


    def __fan_points(self, angle: float, d: float, blt: float) -> float:
        """
        Calculate the amount of points in the fan.
        """
        # fan point mult factor:
        fanang = np.pi - angle
        if fanang < 0:
            return 0
        else:
            return int(self.bl_fdf * fanang * blt * d)


    def __curve_density(self, c: np.ndarray, va: list, fp: np.ndarray) -> np.ndarray:
        """
        Compute the density of a curve.
        """
        lc = len(c)
        # default density
        defd = 1/self.bl_sp
        # orientation
        nrmls = geo.curve.normals(c)
        angs = np.acos(nrmls[:,0])
        pv = np.nonzero(angs > np.pi/2)[0]
        angs = np.abs(angs - np.pi/2)
        coefs = np.full(lc, self.bl_oc[0])
        coefs[pv] = self.bl_oc[1]
        orc = coefs * angs
        # curvature
        crvtr = geo.curve.curvature(c)
        nv = np.nonzero(crvtr < 0)[0]
        crvtr[nv] = -crvtr[nv]
        coefs = np.full(lc, self.bl_cc[0])
        coefs[nv] = self.bl_cc[1]
        crc = coefs * crvtr
        # proximity
        if len(fp) > 0:
            proxm = np.min(geo.distance(c, fp), axis=1)
            prc = self.bl_pc / proxm
        else:
            prc = 0
        # vertex angs
        va = np.array(va)
        nv = np.nonzero(va > np.pi)
        va = np.abs(va - np.pi)
        coefs = np.full(2,self.bl_vsc[0])
        coefs[nv] = self.bl_vsc[1]
        vrac = coefs * va
        # summed coefficients
        s_coef = orc + crc + prc + 1
        s_coef[0] = s_coef[0] + vrac[0]
        s_coef[-1] = s_coef[-1] + vrac[1]
        return defd * s_coef


    def __wake(self, section: geo.Section, afl_dens: list, b_t: list) -> np.ndarray:
        """
        Generate points that increase the mesh density in the wake region.
        """
        # angle limits
        alims = [-np.pi/2 + min(self.w_aoa), np.pi/2 + max(self.w_aoa)]
        profiles = Mesh.__get_profiles(section)
        prof_points = np.vstack(profiles)
        wake_points = np.array([], dtype=float).reshape(0,3)

        # find maximum chord
        maxchord = 0
        for afl in section.shapes:
            if afl.chord > maxchord:
                maxchord = afl.chord

        for i in range(len(section.shapes)):
            # profiles
            native_prof = profiles[i]
            foreign_prof = list(profiles)
            foreign_prof.pop(i)
        
            # remove tiny edges for consistency
            afl = list(section.shapes[i].curves)
            crv_dens = afl_dens[i]

            j = 0
            while j < len(afl):
                if geo.curve.length(afl[j])[-1] < 0.025*maxchord:
                    afl.pop(j)
                    crv_dens.pop(j)
                else:
                    j += 1
            
            afl.insert(0, afl[-1])
            # generate and clean trail for every edge
            for j in range(len(afl)-1):
                wv1 = afl[j][-2] - afl[j][-1]
                wv2 = afl[j+1][1] - afl[j+1][0]
                wv = -geo.vector.bisector(wv1, wv2)
                wva = geo.vector.angle(wv1, wv2)
                if np.any(wv == np.nan) or (wva < 0) or (wva > self.w_mva):
                    continue
                if alims[0] <= geo.vector.angle(wv) <= alims[1]:
                    d = crv_dens[j][-1]
                    p = (afl[j][-1] + afl[j+1][0])/2
                    wtails = self.__wake_tail(wv, p, d)
                    for wtail in wtails:
                        # remove points inside own profile
                        wtail = np.delete(wtail, np.nonzero(geo.inpolyg(wtail[:,0:2], native_prof))[0], axis=0)
                        # cut tail short if it intersects other shapes
                        for prof in foreign_prof:
                            inti = np.nonzero(geo.inpolyg(wtail[:,0:2], prof))[0]
                            if len(inti) > 0:
                                wtail = wtail[0:inti[0]]
                        # cut tail short if it's spacing grows too big, within the inflation layer
                        dists = np.min(geo.distance(wtail[:,0:2], prof_points), axis=1)
                        proxi = np.nonzero(dists < self.il_d)[0]
                        cuti = np.nonzero((wtail[proxi,2] > self.il_nfs))[0]
                        if len(cuti) > 0:
                            wtail = wtail[0:proxi[cuti[0]]]
                            dists = dists[0:proxi[cuti[0]]]
                        # remove points unacceptably close to the boundary layer
                        for k, profile in enumerate(profiles):
                            dists = np.min(geo.distance(wtail[:,0:2], profile), axis=1)
                            deli = np.nonzero(dists < 1.5*(b_t[k] + wtail[:,2]))[0]
                            wtail = np.delete(wtail, deli, axis=0)
            
                        wake_points = np.append(wake_points, wtail, axis=0)

        return wake_points


    def __wake_tail(self, wv: np.ndarray, p: np.ndarray, d: float) -> np.ndarray:
        """
        Generate points of a wake tail of a single edge.
        """
        xfar = self.cv_xr[1]
        ddg = self.w_dd
        tails = []
        for aoa in self.w_aoa:
            p2 = p + geo.vector.unit(wv) * self.w_dr
            p3 = [xfar, np.tan(aoa) * (xfar - p2[0]) + p2[1]]
            bez = geo.spline.bezier([p,p2,p3])
            # get number of points
            bez.delta = 0.005
            l = geo.curve.length(bez.evalpts)[-1]
            nop = 2*int(l/self.w_fs)
            # get points
            pvals = np.linspace(0,1,nop+1)**2
            tailp = bez.evaluate_list(pvals.tolist())
            tailp=np.array(tailp)
            # spacings
            s = np.transpose([(np.linspace((0.2/d)**(1/ddg), self.w_fs**(1/ddg), nop+1)**ddg)])
            tails.append(np.hstack((tailp, s)))
        
        return tails


    def __inflation_layer(self, section: geo.Section, wakep: np.ndarray) -> np.ndarray:
        """
        Generate inflation layers over the profiles of the section.
        """
        df = 1.2
        profiles = Mesh.__get_profiles(section)
        dpf = 0.9
        dtf = 0.4
        profp = np.vstack(profiles)
        inflayers = []
        for prof in profiles:
            inflp = geo.curve.offset(prof, self.il_d)
            # clear inflation points off of section
            dists = geo.distance(inflp, profp)
            deli = np.nonzero(np.min(dists,axis=1) < dpf*self.il_d)[0]
            inflp = np.delete(inflp, deli, axis=0)
            # clear inflation points off of wake
            dists = geo.distance(inflp, wakep)
            deli = np.nonzero(np.min(dists,axis=1) < dtf*self.il_d)[0]
            inflp = np.delete(inflp, deli, axis=0)
            # clear inflation points off of each other
            dists = geo.distance(inflp, inflp) + np.diagflat(np.full(len(inflp), np.inf))
            inflpc = []
            while len(inflp) > 0:
                mdi = np.argmin(dists[0])
                if dists[0,mdi] < df * self.il_nfs:
                    inflp = np.delete(inflp, mdi, axis=0)
                    dists = np.delete(dists, mdi, axis=0)
                    dists = np.delete(dists, mdi, axis=1)
                else:
                    inflpc.append(inflp[0])
                    inflp = np.delete(inflp, 0, axis=0)
                    dists = np.delete(dists, 0, axis=0)
                    dists = np.delete(dists, 0, axis=1)

            inflp = np.array(inflpc)
            if len(inflp) > 0:
                s = np.full((len(inflp),1), self.il_nfs)
                inflayers.append(np.hstack((inflp, s)))
        
        return np.vstack(inflayers)
