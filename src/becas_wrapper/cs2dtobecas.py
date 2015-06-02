__all__ = ['BECASWrapper']

import os
import time
import numpy as np
import logging
from openmdao.main.api import Component
from openmdao.lib.datatypes.api import (Int, Float, Str, List, Array, Dict,
                                        VarTree, Bool)

from fusedwind.turbine.structure_vt import Region, CrossSectionStructureVT


class CS2DtoBECAS(Component):
    """
    Component that generates a set of BECAS input files based on
    a fusedwind.turbine.structure_vt.CrossSectionStructureVT.

    This class uses shellexpander, which comes as part of the BECAS distribution.
    """

    path_shellexpander = Str(iotype='in', desc='Absolute path to shellexpander.py')

    cs2d = VarTree(CrossSectionStructureVT(), iotype='in', desc='Cross-sectional properties vartree')
    total_points = Int(100, iotype='in')
    open_te = Bool(False, iotype='in', desc='If True, TE will be left open')
    becas_inputs = Str('becas_inputs', iotype='in',
                       desc='Relative path for the future BECAS input files')
    section_name = Str('BECAS_SECTION', iotype='in',
                       desc='Section name used by shellexpander, also by BECASWrapper')
    dominant_elsets = List(['REGION03', 'REGION06'], iotype='in',
                           desc='define the spar cap regions for correct meshing')

    path_input = Str(iotype='out', desc='path to the generated BECAS input files')
    airfoil = Array(iotype='out', desc='the re-distributed airfoil coordinates')
    web_coord = Array(iotype='out', desc='the distributed shear web coordinates')
    nodes = Array(iotype='out',
                  desc='all the nodes, web+airfoil for the 2D cross section')
    elset_defs = Dict(iotype='out', desc='')
    elements = Array(iotype='out', desc='')
    nodes_3d = Array(iotype='out', desc='')
    el_3d = Array(iotype='out', desc='')

    def execute(self):
        """  """

        self._logger.info('starting CS2DtoBECAS ...')
        if self.path_shellexpander == '':
            raise RuntimeError('path_shellexpander not specified')

        tt = time.time()

        self.path_input = os.path.join(self.becas_inputs, self.section_name)

        if self.cs2d.airfoil.points.shape[0] > 0:
            self.iCPs = []
            self.WCPs = []
            self.iWCPs = []
            self.CPs = []

        self.total_points_input = self.total_points

        try:
            self.compute_max_layers()
            self.compute_airfoil()
            # self.flatback()
            self.adjust_te_panels()

            self.add_shearweb_nodes()
            self.create_elements()
            self.create_elements_3d(reverse_normals=False)
            self.write_abaqus_inp()
            self.write_becas_inp()
        except:
            self._logger.info('CS2DtoBECAS failed building BECAS mesh')

        self._logger.info('CS2DtoBECAS calc time: % 10.6f seconds' % (time.time()-tt))

    def compute_max_layers(self):
        """
        The number of elements used to discretize the shell thickness.
        The minimum value is the largest number of layers of different
        material anywhere in the airfoil.
        """

        self.max_layers = 0
        self.max_thickness = 0.
        for r_name in self.cs2d.regions:
            r = getattr(self.cs2d, r_name)
            self.max_layers = max(self.max_layers, len(r.layers))
            
            if r.thickness == 0.:
                # in case the total layer thickness for each region is not already computed
                for l_name in r.layers:
                    lay = getattr(r, l_name)
                    r.thickness += lay.thickness
            if r.thickness > self.max_thickness: self.max_thickness = r.thickness

    def compute_airfoil(self, debug=False):
        """Redistributed mesh points evenly among regions

        After defining different regions this method will assure that, given
        a total number of mesh points, the cell size in each region is similar.
        The region boundaries and shear web positions will be approximately
        on the same positions as defined.

        Region numbers can be added arbitrarely.
        """

        af = self.cs2d.airfoil

        if np.linalg.norm(self.cs2d.airfoil.points[0] - self.cs2d.airfoil.points[-1]) > 0.:
            self.open_te = True

        # construct distfunc for airfoil curve
        dist = []

        # the distribution function only cares about control points and does
        # not have any notion of the regions. Prepare a sorted list for all
        # the control points, including the webs
        # add first point already: the TE
        CPs_s, WCPs_s = [-1], []
        self.iCPs.append(0)

        # because we want an evenly distributed mesh, calculate the cell
        # size note that distfunc ds does not relate to self._af.smax for
        # its length
        ds_const = 1.0 / self.total_points
        if ds_const * af.length < 1.2 * self.max_thickness:
            ds_old = ds_const
            new_ds = 1.2 * self.max_thickness
            self.total_points = np.maximum(int(af.length / new_ds), 70)
            ds_const = 1. / self.total_points
            self._logger.info('Increasing cell size from %5.3f to %5.3f '
                'and reducing number of elements to %i' % (ds_const, ds_old, self.total_points))
        self.ds_const = ds_const
        # add first point, TE, s0=0 in AirfoilShape curve fraction coordinates
        # dist.append([0.0, ds_const, 1])
        self.CPs.append(af.interp_s(af.s_to_01(-1.)))
        # keep track of all CPs and WCPs in one dictionary
        allCPs = {}
        # cycle through all the regions
        for name in self.cs2d.regions:
            r = getattr(self.cs2d, name)
            # save the coordinates of the CPs that served as inputs
            self.CPs.append(af.interp_s(af.s_to_01(r.s1)))
            # save to the dictionary for sorting
            if r.s1 in CPs_s:
                raise UserWarning, "Each CP's s1 value should be unique"
            CPs_s.append(r.s1)
            allCPs[r.s1] = r
        # and now the webs, if any
        for name in self.cs2d.webs:
            w = getattr(self.cs2d, name)
            if w.thickness == 0.: continue

            WCPs_s.append(w.s0)
            WCPs_s.append(w.s1)
            # save the coordinates of the WCPs that served as inputs
            self.WCPs.append(af.interp_s(af.s_to_01(w.s0)))
            self.WCPs.append(af.interp_s(af.s_to_01(w.s1)))
            # web control points are allowed to coincide with CP's. If so,
            # there is no need to add another control point
            if w.s0 not in allCPs:
                allCPs[w.s0] = w
            if w.s1 not in allCPs:
                allCPs[w.s1] = w

        # now sort the list so we can properly construct a Curve
        dist_ni = 0
        sorted_allCPs_keys = sorted(allCPs.keys())
        for r_nr, s in enumerate(sorted_allCPs_keys):
            r = allCPs[s]
            # keep track of region start and ending points seperately since
            # webs can interupt the regions
            s_end = s
            if r_nr == 0:
                # the starting point of the curve was already added
                s_start = -1
            else:
                # start is the end necesarely with the previous region,
                # because a WEB doesn't have a start. Consequently, just rely
                # on the sorted allCPs list.
                s_start = sorted_allCPs_keys[r_nr-1]

            # only set the starting index of a region of the current point
            # actually belongs to region and not a shear web
            if not r.name.lower().startswith('web'):
                # explicitaly take the previous region end point
                r.s0_i = self.iCPs[-1]

            if debug:
                print '% 6.3f  % 6.3f  %8s' % (s_start, s_end, r.name)

            # for AirfoilShape we want surface fractions between 0 and 1
            s_start_ = af.s_to_01(s_start)
            s_end_ = af.s_to_01(s_end)

            # find the amount of points required for this region
            # and the dist_ni refers to the total number of nodes on the
            # curve when at the end of the region (s1)
            dist_ni += max(1, int(round( (s_end_ - s_start_)/ds_const )))
            # add a distribution point to the Curve
            dist.append([s_end_, ds_const, dist_ni])

            # save the index of the end point of the region
            # is it a region, or s0 or s1 of a web?
            if r.name.lower().startswith('web'):
                if s_end == r.s0:
                    r.s0_i = dist_ni-1
                elif s_end == r.s1:
                    r.s1_i = dist_ni-1
            else:
                # some CPs might coincide with WCPs, in that case it would
                # not have been in allCPs
                if s_end in WCPs_s:
                    # now figure out to which region the current s_end belongs
                    for w_name in self.cs2d.webs:
                        w = getattr(self.cs2d, w_name)
                        if s_end == w.s0:
                            w.s0_i = dist_ni-1
                        elif s_end == w.s1:
                            w.s1_i = dist_ni-1
                # but also still add s1_i to the region object
                r.s1_i = dist_ni-1
            # and save the index for later reference in a convienent list
            if r.s1 in CPs_s:
                self.iCPs.append(dist_ni-1)
            # be adviced, a CP can also live on the web
            if r.s1 in WCPs_s:
                self.iWCPs.append(dist_ni-1)

        # before executing, make sure all the points are in increasing order
        if np.diff(np.array(dist)[:, 0]).min() <= 0:
            raise ValueError, 'Points are not continiously increasing'
        afn = af.redistribute(dist_ni, dist=dist)
        # get the redistributed points
        self.airfoil = afn.points
        self.total_points = self.airfoil.shape[0]
        self.CPs_s = CPs_s
        self.WCPs_s = WCPs_s

    def mirror_airfoil(self):
        """
        mirror the airfoil, multiply x axis with -1
        """

        offset = 0.0

        self.web_coord[:,0] *= -1.0
        self.airfoil[:,0] *= -1.0

        # also for the control points
        for cp in self.CPs:
            cp[0] *= -1.0
            cp[0] += offset
        for wcp in self.WCPs:
            wcp[0] *= -1.0
            wcp[0] += offset

        if self.te_le_orientation == 'right-to-left':
            self.te_le_orientation = 'left-to-right'
        else:
            self.te_le_orientation = 'right-to-left'

    def add_shearweb_nodes(self):
        """
        Distribute nodes over the shear web. Use the same spacing as used for
        the airfoil nodes.
        """

        self.nr_webs = 0
        self.web_coord = np.array([])
        for w_name in self.cs2d.webs:
            w = getattr(self.cs2d, w_name)
            if w.thickness == 0.: continue
            self.nr_webs += 1
            # at this point there are no nodes on the shear web itself
            # add a node distribution on the shear web too, and base the node
            # spacing on the average spacing ds on the airfoil
            # TODO: should'nt the shear web elements be assigned in
            # compute_airfoil?
            ds_mean = np.maximum(self.ds_const * self.cs2d.airfoil.length, self.max_thickness * 1.2)
            node1 = self.airfoil[w.s0_i,:]
            node2 = self.airfoil[w.s1_i,:]
            # the length of the shear web is then
            len_web = np.linalg.norm( node1-node2 )
            nr_points = max(int(round(len_web / ds_mean, 0)), 3)
            # generate nodal coordinates on the shear web
            x = np.linspace(node1[0], node2[0], nr_points)
            y = np.linspace(node1[1], node2[1], nr_points)
            # and add them to the shear web node collection, but ignore the
            # first and last nodes because they are already included in
            # the airfoil coordinates.
            # For small arrays this is slightly faster, but for big arrays
            # (which is already true for 30 items and up) it is better to first
            # create them instead of transposing
            # tmp = np.array([x[1:-1], y[1:-1]]).transpose()
            tmp = np.ndarray((len(x)-2, 2))
            tmp[:,0] = x[1:-1]
            tmp[:,1] = y[1:-1]
            # remember to start and stop indices for the shear web nodes
            w.w0_i = len(self.web_coord)
            w.w1_i = len(tmp) + w.w0_i - 1
            try:
                self.web_coord = np.append(self.web_coord, tmp, axis=0)
            except:
                self.web_coord = tmp.copy()

    def adjust_te_panels(self):
        """
        adjust the thickness of the trailing edge panels according
        to the thickness of the trailing edge

        """

        if not self.open_te: return

        # pressure and suction side panels
        dTE = np.abs(self.airfoil[-1, 1] - self.airfoil[0, 1]) / 3.
        r_name = self.cs2d.regions[-1]
        r_TE_suc = getattr(self.cs2d, r_name)
        r_name = self.cs2d.regions[0]
        r_TE_pres = getattr(self.cs2d, r_name)
        thick_max = (r_TE_pres.thickness + r_TE_suc.thickness) / 2.
        ratio = thick_max / dTE
        self._logger.info('TE panel ratio %f %f %f' % (self.cs2d.s, dTE * 3., ratio))
        if ratio > 1.:
            for lname in r_TE_suc.layers:
                layer = getattr(r_TE_suc, lname)
                layer.thickness = layer.thickness / ratio
            for lname in r_TE_pres.layers:
                layer = getattr(r_TE_pres, lname)
                layer.thickness = layer.thickness / ratio
            r_TE_suc.thickness /= ratio
            r_TE_pres.thickness /= ratio

        # trailing edge "web"
        for name in self.cs2d.webs:
            TEw = getattr(self.cs2d, name)
            if TEw.s0 in [-1., 1.]:
                dTE = dTE * 2.
                ratio = r_TE_suc.thickness / TEw.thickness
                for lname in TEw.layers:
                    layer = getattr(TEw, lname)
                    layer.thickness = layer.thickness * ratio
                TEw.thickness *= ratio
                break

    def flatback(self):
        """
        Instead of removing some meshing points, make the TE region as thick
        as the total defined layer thickness in that region.
        """

        # find the tickness in the TE region: first and last region
        r_name = self.cs2d.regions[-1]
        r_TE_suc = getattr(self.cs2d, r_name)
        r_name = self.cs2d.regions[0]
        r_TE_pres = getattr(self.cs2d, r_name)
        # add 10% margin as well for safety
        thick_max = (r_TE_pres.thickness + r_TE_suc.thickness)*1.1

        # and enforce that thickness on the trailing edge node suction side
        # first, define the trailing edge vector.
        if np.allclose(self.airfoil[-1,:], self.airfoil[0,:]):
            # when TE suction = TE pressure, move upwards vertically
            flatback_thick = 0
            flatback_vect_norm = np.array([0,1])
        else:
            flatback_vect = self.airfoil[-1,:] - self.airfoil[0,:]
            flatback_thick = np.linalg.norm(flatback_vect)
            flatback_vect_norm = flatback_vect/flatback_thick
        if flatback_thick < thick_max:
            dt_thick = thick_max - flatback_thick
            # add missing thickness by moving the TE suction side upwards
            # along the flatback vector
            self.airfoil[-1, :] += dt_thick*flatback_vect_norm * 0.5
            self.airfoil[0, :] -= dt_thick*flatback_vect_norm * 0.5

    def _check_TE_thickness(self):
        """
        The last point before the trailing edge should still have a thickness
        that is equal or higher than the total layer thickness (suction and
        pressure side combined).

        Two solutions: either reduce the layer thickness, or more simple,
        move the last point before the TE forward (direction of LE) so the
        thickness over that mesh point increases.

        This method looks for the point where the layer thickness equals the
        chord thickness and takes that as the last mesh point before the TE.

        Possible problems arise if a whole region needs to be cut down.
        Also, all the indices in the regions have to be changed...
        From here onwards the two meshes will be disconnected.
        This approach will cause problems if there still is significant
        curvature in this trailing edge area. Maybe a boundary condition
        is more appropriate?
        Possibly verify if the reduced number of points results in a loss
        of area accuracy compared to the original input coordinates
        """

        # find the local tickness of the last trailing edge points
        # TE region is here defined as 15% of the chord
        # FIXME: this approach assumes the thickness can be approximated by
        # the distance between the mesh points with equal index offset from
        # the TE. However, this is by no means guaranteed by the mesh, and will
        # result in some cases a much higher percieved thickness in the TE.
        nr = int(round(len(self.airfoil)*0.15,0))
        # calculate the thickness at each pair of nodes
        deltas = self.airfoil[1:nr] - self.airfoil[-nr:-1][::-1]
        if np.__version__ >= '1.8.0':
            thick_loc = np.linalg.norm(deltas, axis=1)
        else:
            thick_loc = np.ndarray( (deltas.shape[0],) )
            for i, d in enumerate(deltas):
                thick_loc[i] = np.linalg.norm(d)

        # find the tickness in the TE region: first and last region
        r_name = self.cs2d.regions[-1]
        r_TE_suc = getattr(self.cs2d, r_name)
        r_name = self.cs2d.regions[0]
        r_TE_pres = getattr(self.cs2d, r_name)
        # add 10% margin as well
        thick_max = (r_TE_pres.thickness + r_TE_suc.thickness)*1.1

        # TODO: before removing we should check what happens if we by accident
        # remove a control point and/or a complete region

        # and see how many nodes we need to ditch in the TE area
        # delete one extra node just to be sure
        nr_remove = thick_loc.__gt__(thick_max).argmax() + 1
        sel = np.ndarray( (self.airfoil[0],), dtype=np.bool)
        sel[:] = True
        sel[1:nr_remove+1] = False
        sel[-nr_remove-1:-1] = False
        self.airfoil = self.airfoil[sel,:]

        print "number of removed mesh points at TE: %i" % nr_remove

        # and correct all the indices
        for name in self.cs2d.regions:
            r = getattr(self, name)
            if r.s0_i >= nr_remove:
                r.s0_i -= nr_remove
                r.s1_i -= nr_remove
            elif r.s1_i >= nr_remove:
                r.s1_i -= nr_remove
        # additionally, the TE on the suction side also loses on top of that
        r = getattr(self, self.cs2d.regions[-1])
        r.s1_i -= nr_remove

        for name in self.webs:
            w = getattr(self.cs2d.webs, name)
            w.s0_i -= nr_remove
            w.s1_i -= nr_remove
        for i in xrange(len(self.iWCPs)):
            self.iWCPs[i] -= (nr_remove)
        # first and last are the TE
        for i in xrange(len(self.iCPs[1:])):
            self.iCPs[i+1] -= (nr_remove)

        # TODO: this should follow the same procedure as for the regions
        self.iCPs[-1] -= nr_remove

    def create_elements(self, debug=False):
        """
        Create the elements and assign element sets to the different regions.

        Assign node and element numbers for the current airfoil points and
        shear webs. Since the airfoil coordinates are ordered clockwise
        continuous the node and element numbering is trivial.

        Note when referring to node and element numbers array indices are used.
        BECAS uses 1-based counting instead of zero-based.
        """

        # by default, the node and element numbers are zero based numbering
        self.onebasednumbering = False

        # convert to BECAS standards if necessary
        # if self.te_le_orientation == 'right-to-left':
        #     self.mirror_airfoil()
        #     print 'forced airfoil input coordinates orientation left-to-right'

        # element definitions for 1D line elements
        # line_element_definitions = {}
        # corresponds to self.elements (index is the element number)

        # element numbers for each ELSET
        self.elset_defs = {}

        nr_air_n = len(self.airfoil)
        nr_web_n = len(self.web_coord)
        nr_nodes = nr_air_n + nr_web_n
        # for closed TE, nr_elements = nr_nodes, for open TE, 1 element less
        if self.open_te:
            nr_elements = nr_nodes + len(self.cs2d.webs) - 1
            nr_air_el = nr_air_n - 1
        else:
            nr_elements = nr_nodes + len(self.cs2d.webs)
            nr_air_el = nr_air_n

        # place all nodal coordinates in one array. The elements are defined
        # by the node index.
        self.nodes = np.zeros( (nr_nodes, 3) )
        self.nodes[:nr_air_n,:2] = self.airfoil[:,:]
        self.nodes[nr_air_n:,:2] = self.web_coord

        # Elements are bounded by two neighbouring nodes. By closing the
        # circle (connecting the TE suction side with the pressure side), we
        # have as many elements as there are nodes on the airfoil
        # elements[element_nr, (node1,node2)]: shape=(n,2)
        # for each web, we have nr_web_nodes+1 number of elements
        self.elements = np.ndarray((nr_elements, 2), dtype=np.int)
        if self.open_te:
            self.elements[:nr_air_el,0] = np.arange(nr_air_n-1, dtype=np.int)
            self.elements[:nr_air_el,1] = self.elements[:nr_air_el,0] + 1
        else:
            # when the airfoil is closed, add one node number too much...
            self.elements[:nr_air_el,0] = np.arange(nr_air_n, dtype=np.int)
            self.elements[:nr_air_el,1] = self.elements[:nr_air_el,0] + 1
            # last node on last element is first node, airfoil is now closed
            self.elements[nr_air_el-1,1] = 0

        if debug:
            print 'nr airfoil nodes: %4i' % (len(self.airfoil))
            print '    nr web nodes: %4i' % len(self.web_coord)

        web_el = []
        pre_side, suc_side = [], []

        # keep track of elements that have been added on the shear webs
        el_offset = 0
        # define el for each shear web, and create corresponding node groups
        for w_name in self.cs2d.webs:
            # starting index for the elements of the web
            iw_start = nr_air_el + el_offset

            w = getattr(self.cs2d, w_name)
            w.is_TE = False
            if w.thickness == 0.:continue

            # number of intermediate shear web elements (those that are not
            # connected to the airfoil)
            nr_el = w.w1_i - w.w0_i

            # define start/stop element indices
            w.e0_i = iw_start
            w.e1_i = nr_el + iw_start + 1

            # shear web nodes run from w.s0 towards w.s1
            # first element is connected to the starting shear web point that
            # sits on the airfoil: nr_el+1
            self.elements[w.e0_i,:] = [w.s0_i, w.w0_i + nr_air_n]

            # elements in between
            wnodes = np.arange(w.w0_i, w.w1_i, dtype=np.int) + nr_air_n
            self.elements[w.e0_i+1:w.e1_i, 0] = wnodes
            self.elements[w.e0_i+1:w.e1_i, 1] = wnodes + 1
            # and the final element that connects the web back to the airfoil
            # nr_el+2
            self.elements[w.e1_i, :] = [w.w1_i+nr_air_n, w.s1_i]

            if debug:
                print '%s i_el start: %i' % (w_name, iw_start)
                print '%4i %4i %4i' % (iw_start, w.s0_i, w.w0_i+nr_air_n)
                print '%4i %4i %4i' % (w.e1_i, w.w1_i+nr_air_n, w.s1_i)

            # and now we can populate the different regions with their
            # corresponding elements
            if w.s0 in [-1., 1.]:
                w.is_TE = True
                self.elset_defs[w_name] = np.array([w.e0_i, w.e1_i] + [0], dtype=int)
                suc_side.extend([w.e0_i, w.e1_i+1] + [1])
                self._logger.info('TE web identified! %s %i %i' % (w_name, w.s0_i, w.s1_i))
            else:
                self.elset_defs[w_name] = np.arange(w.e0_i, w.e1_i+1, dtype=np.int)
                web_el.extend(range(w.e0_i, w.e1_i+1))

            # add the number of elements added for this web
            el_offset += nr_el+2

        if len(web_el) > 0:
            self.elset_defs['WEBS'] = np.array(web_el, dtype=np.int)

        # element groups for the regions
        for r_name in self.cs2d.regions:
            r = getattr(self.cs2d, r_name)
            # do not include element r.s1_i, that is included in the next elset
            self.elset_defs[r_name] = np.arange(r.s0_i, r.s1_i, dtype=np.int)

            # group in suction and pressure side (s_LE=0)
            if r.s1 <= 0:
                pre_side.extend([r.s0_i, r.s1_i])
            else:
                suc_side.extend([r.s0_i, r.s1_i])

        tmp = np.array(list(pre_side)+list(suc_side))
        pre0, pre1 = tmp.min(), tmp.max()
        self.elset_defs['SURFACE'] = np.arange(pre0, pre1, dtype=np.int)

        # the last region and the suction side do not include the last element
        # for flatback airfoils. Fix this here.
        # TODO: now the region object s1_i is not updated. Is that relevant?
        # Or could that brake things since the index start-stop is not increasing
        # if not self.open_te:
        #     r_name = self.cs2d.regions[-1]
        #     r = getattr(self.cs2d, r_name)
        #     # last region is the suction side trailing edge
        #     elset = self.elset_defs[r_name]
        #     self.elset_defs[r_name] = np.append(elset, np.array([nr_air_n-1]))
        #     # suction side
        #     elset = self.elset_defs['SUCTION_SIDE']
        #     self.elset_defs['SUCTION_SIDE'] = np.append(elset,
        #                                                 np.array([nr_air_n-1]))

    def create_elements_3d(self, reverse_normals=False):
        """
        Shellexpander wants a 3D section as input. Create a 3D section
        which is just like the 2D version except with a depth defined as 1%
        of the chord length.
        """

        # Compute depth of 3D mesh as 1% of chord lenght
        depth = -0.01 * self.cs2d.airfoil.chord
        if reverse_normals:
            depth = depth * (-1.0)

        nr_nodes_2d = len(self.nodes)
        # Add nodes for 3D mesh
        self.nodes_3d = np.ndarray( (nr_nodes_2d*2, 3) )
        self.nodes_3d[:nr_nodes_2d, :] = self.nodes
        self.nodes_3d[nr_nodes_2d:, :] = self.nodes + np.array([0,0,depth])
        # Generate shell elements
        self.el_3d = np.ndarray( (len(self.elements), 4), dtype=np.int)
        self.el_3d[:,:2] = self.elements
        self.el_3d[:,2] = self.elements[:,1] + nr_nodes_2d
        self.el_3d[:,3] = self.elements[:,0] + nr_nodes_2d

        # same as line_element_definitions, but now expanded over thickness
        # to create a 3D shell element
        # element_definitions = {}
        # corresponds to self.el_3d

    def one_based_numbering(self):
        """
        instead of 0, 1 is the first element and node number. All nodes and
        elements +1.

        Note that this does not affect the indices defined in the region
        and web attributes
        """
        if not self.onebasednumbering:

            self.elements += 1
            self.el_3d += 1
            for elset in self.elset_defs:
                self.elset_defs[elset] += 1

            for name in self.cs2d.regions:
                r = getattr(self.cs2d, name)
                r.s0_i += 1
                r.s1_i += 1
            # webs refer to indices in self.web_coord, which is independent
            # of self.airfoil
#            for name in self._webs:
#                w = getattr(self, name)
#                w.s0_i += 1
#                w.s1_i += 1
            for i in xrange(len(self.iWCPs)):
                self.iWCPs[i] += 1
            for i in xrange(len(self.iCPs)):
                self.iCPs[i] += 1

            self.onebasednumbering = True

    def zero_based_numbering(self):
        """
        switch back to 0 as first element and node number
        """

        if self.onebasednumbering:

            self.elements -= 1
            self.el_3d -= 1
            for elset in self.elset_defs:
                self.elset_defs[elset] -= 1

            for name in self.cs2d.regions:
                r = getattr(self, name)
                r.s0_i -= 1
                r.s1_i -= 1
            # webs refer to indices in self.web_coord, which is independent
            # of self.
#            for name in self._webs:
#                w = getattr(self, name)
#                w.s0_i -= 1
#                w.s1_i -= 1
            for i in xrange(len(self.iWCPs)):
                self.iWCPs[i] -= 1
            for i in xrange(len(self.iCPs)):
                self.iCPs[i] -= 1

            self.onebasednumbering = False

    def check_airfoil(self):

        for rname in self.cs2d.regions:
            r = getattr(self.cs2d, rname)
            print 'Checking %s' % rname
            for lname in r.layers:
                print '    Checking %s' % lname
                l = getattr(r, lname)
                if l.thickness <= 0.:
                    print 'ERROR! Layer %s in Region %s has negative thickness: %f' % (lname, rname, l.thickness)
        for rname in self.cs2d.webs:
            print 'Checking %s' % rname
            r = getattr(self.cs2d, rname)
            for lname in r.layers:
                print '    Checking %s' % lname
                l = getattr(r, lname)
                if l.thickness <= 0.:
                    print 'ERROR! Layer %s in Region %s has negative thickness: %f' % (lname, rname, l.thickness)

    def write_abaqus_inp(self, fname=False):
        """Create Abaqus inp file which will be served to shellexpander so
        the actual BECAS input can be created.
        """

        def write_n_int_per_line(list_of_int, f, n):
            """Write the integers in list_of_int to the output file - n integers
            per line, separated by commas"""
            i=0
            for number in list_of_int:
                i=i+1
                f.write('%d' %(number ))
                if i < len(list_of_int):
                    f.write(',  ')
                if i%n == 0:
                    f.write('\n')
            if i%n != 0:
                f.write('\n')

        self.abaqus_inp_fname = 'airfoil_abaqus.inp'

        # FIXME: for now, force 1 based numbering, I don't think shellexpander
        # and/or BECAS like zero based node and element numbering
        self.one_based_numbering()

        # where to start node/element numbering, 0 or 1?
        if self.onebasednumbering:
            off = 1
        else:
            off = 0

        with open(self.abaqus_inp_fname, 'w') as f:

            # Write nodal coordinates
            f.write('**\n')
            f.write('********************\n')
            f.write('** NODAL COORDINATES\n')
            f.write('********************\n')
            f.write('*NODE\n')
            tmp = np.ndarray( (len(self.nodes_3d),4) )
            tmp[:,0] = np.arange(len(self.nodes_3d), dtype=np.int) + off
            tmp[:,1:] = self.nodes_3d
            np.savetxt(f, tmp, fmt='%1.0f, %1.8e, %1.8e, %1.8e')

            # Write element definitions
            f.write('**\n')
            f.write('***********\n')
            f.write('** ELEMENTS\n')
            f.write('***********\n')
            f.write('*ELEMENT, TYPE=S4, ELSET=%s\n' % self.section_name)
            tmp = np.ndarray( (len(self.el_3d),5) )
            tmp[:,0] = np.arange(len(self.el_3d), dtype=np.int) + off
            tmp[:,1:] = self.el_3d
            np.savetxt(f, tmp, fmt='%i, %i, %i, %i, %i')

            # Write new element sets
            f.write('**\n')
            f.write('***************\n')
            f.write('** ELEMENT SETS\n')
            f.write('***************\n')
            for elset in sorted(self.elset_defs.keys()):
                elements = self.elset_defs[elset]
                f.write('*ELSET, ELSET=%s\n' % (elset))
#                np.savetxt(f, elements, fmt='%i', delimiter=', ')
                write_n_int_per_line(list(elements), f, 8)

            # Write Shell Section definitions
            # The first layer is the outer most layer.
            # The second item ("int. points") and the fifth item ("plyname")
            # are not relevant. The are kept for compatibility with the ABAQUS
            # input syntax. As an example, take this one:
            # [0.006, 3, 'TRIAX', 0.0, 'Ply01']
            f.write('**\n')
            f.write('****************************\n')
            f.write('** SHELL SECTION DEFINITIONS\n')
            f.write('****************************\n')
            for i, r_name in enumerate(self.cs2d.regions + self.cs2d.webs):
                r = getattr(self.cs2d, r_name)
                text = '*SHELL SECTION, ELSET=%s, COMPOSITE, OFFSET=-0.5\n'
                f.write(text % (r_name))
                for l_name in r.layers:
                    lay = getattr(r, l_name)
                    md = self.cs2d.materials[lay.materialname.lower()]
                    if md.failure_criterium == 'maximum_stress':
                        mname = lay.materialname + 'MAXSTRESS'
                    elif md.failure_criterium == 'maximum_strain':
                        mname = lay.materialname + 'MAXSTRAIN'
                    elif md.failure_criterium == 'tsai_wu':
                        mname = lay.materialname + 'TSAIWU'
                    else:
                        mname = lay.materialname
                    if lay.plyname == '': lay.plyname = 'ply%02d' % i

                    layer_def = (lay.thickness, 3, mname,
                                 lay.angle, lay.plyname)
                    f.write('%g, %d, %s, %g, %s\n' % layer_def )

            # Write material properties
            f.write('**\n')
            f.write('**********************\n')
            f.write('** MATERIAL PROPERTIES\n')
            f.write('**********************\n')
            for matname in sorted(self.cs2d.materials.keys()):
                md = self.cs2d.materials[matname]
                if md.failure_criterium == 'maximum_stress':
                    mname = matname + 'MAXSTRESS'
                elif md.failure_criterium == 'maximum_strain':
                    mname = matname + 'MAXSTRAIN'
                elif md.failure_criterium == 'tsai_wu':
                    mname = matname + 'TSAIWU'
                else:
                    mname = matname
                f.write('*MATERIAL, NAME=%s\n' % (mname))
                f.write('*ELASTIC, TYPE=ENGINEERING CONSTANTS\n')
                f.write('%g, %g, %g, %g, %g, %g, %g, %g\n' % (md.E1, md.E2,
                    md.E3, md.nu12, md.nu13, md.nu23, md.G12, md.G13))
                f.write('%g\n' % (md.G23))
                f.write('*DENSITY\n')
                f.write('%g\n' % (md.rho))
                f.write('*FAIL STRESS\n')
                gMa = md.gM0 * (md.C1a + md.C2a + md.C3a + md.C4a)
                f.write('%g, %g, %g, %g, %g\n' % (gMa * md.s11_t, gMa * md.s11_c,
                                                  gMa * md.s22_t, gMa * md.s22_c, gMa * md.t12))
                f.write('*FAIL STRAIN\n')
                f.write('%g, %g, %g, %g, %g\n' % (gMa * md.e11_t, gMa * md.e11_c,
                                                  gMa * md.e22_t, gMa * md.e22_c, gMa * md.g12))
                f.write('**\n')
        print 'Abaqus input file written: %s' % self.abaqus_inp_fname

    def write_becas_inp(self):
        """
        When write_abaqus_inp has been executed we have the shellexpander
        script that can create the becas input

        Dominant regions should be the spar caps.
        """

        class args: pass
        # the he name of the input file containing the finite element shell model
        args.inputfile = self.abaqus_inp_fname #--in
        # The element sets to be considered (required). If more than one
        # element set is given, nodes appearing in more the one element sets
        # are considered "corners". Should be pressureside, suction side and
        # the webs
        elsets = []
        target_sets = ['SURFACE', 'WEBS']
        for elset in target_sets:
            if elset in self.elset_defs:
                elsets.append(elset)
        if len(elsets) < 1:
            raise ValueError, 'badly defined element sets'
        args.elsets = elsets #--elsets, list
        args.sections = self.section_name #--sec
        args.layers = self.max_layers #--layers
        args.nodal_thickness = 'min' #--ntick, choices=['min','max','average']
        # TODO: take the most thick regions (spar caps) as dominant
        args.dominant_elsets = self.dominant_elsets #--dom, list
        args.centerline = None #--cline, string
        args.becasdir = self.becas_inputs #--bdir
        args.debug = False #--debug, if present switch to True

        import imp
        shellexpander = imp.load_source('shellexpander', 
                          os.path.join(self.path_shellexpander, 'shellexpander.py'))

        shellexpander.main(args)

    def plot_airfoil(self, ax, line_el_nr=True):
        """
        """

#        if self.te_le_orientation == 'left-to-right':
#            self.mirror_airfoil()

        nr = self.total_points_input
        points_actual = self.airfoil.shape[0] + self.web_coord.shape[0]
        title = '%i results in %i actual points' % (nr, points_actual)
        ax.set_title(title)
        # the original coordinates from the file
        ax.plot(self.cs2d.airfoil.points[:,0], self.cs2d.airfoil.points[:,1],
                'b--o',  alpha=0.3, label='airfoil')
        ax.plot(self.airfoil[:,0],self.airfoil[:,1],
                'k-s', mfc='k', label='distfunc', alpha=0.3, ms=10)

        # plot all (web)control points
        ax.plot(np.array(self.CPs)[:,0], np.array(self.CPs)[:,1],
                'rx', markersize=7, markeredgewidth=1.2, label='CPs')
        ax.plot(np.array(self.WCPs)[:,0], np.array(self.WCPs)[:,1],
                'gx', markersize=7, markeredgewidth=1.2, label='WCPs')

        # where to start node/element numbering, 0 or 1?
        if self.onebasednumbering:
            off = 1
        else:
            off = 0

        # see if the indices to the control points are what we think they are
        iCPs = np.array(self.iCPs, dtype=np.int) - off
        iWCPs = np.array(self.iWCPs, dtype=np.int) - off
        x, y = self.airfoil[iCPs,0], self.airfoil[iCPs,1]
        ax.plot(x, y,'r+', markersize=12, markeredgewidth=1.2, label='iCPs')
        x, y = self.airfoil[iWCPs,0], self.airfoil[iWCPs,1]
        ax.plot(x, y,'g+', markersize=12, markeredgewidth=1.2, label='iWCPs')

        # plot the webs
        for iweb, w_name in enumerate(self.cs2d.webs):
            w = getattr(self.cs2d, w_name)
            webx = [self.airfoil[self.iWCPs[iweb] - off,0]]
            weby = [self.airfoil[self.iWCPs[iweb] - off,1]]
            webx.extend(self.web_coord[w.w0_i:w.w1_i+1,0])
            weby.extend(self.web_coord[w.w0_i:w.w1_i+1,1])
            webx.append(self.airfoil[self.iWCPs[-iweb-1] - off,0])
            weby.append(self.airfoil[self.iWCPs[-iweb-1] - off,1])
            ax.plot(webx, weby, 'g.-')
#            ax.plot(self.web_coord[w.w0_i:w.w1_i+1,0],
#                    self.web_coord[w.w0_i:w.w1_i+1,1], 'g.-')

        ## add the element numbers
        #print 'nr airfoil nodes: %i' % len(b.airfoil)
        #print '    nr web nodes: %i' % len(b.web_coord)
        #print 'el_nr  node1 node2'
        #for nr, el in enumerate(b.elements):
        #    print '%5i  %5i %5i' % (nr, el[0], el[1])
        bbox = dict(boxstyle="round", alpha=0.8, edgecolor=(1., 0.5, 0.5),
                    facecolor=(1., 0.8, 0.8),)

        # verify all the element numbers
        if line_el_nr:
            for nr, el in enumerate(self.elements):
                # -1 to account for 1 based element numbers instead of 0-based
                p1, p2 = self.nodes[el[0]-off,:], self.nodes[el[1]-off,:]
                x, y, z = (p1+p2)/2.0
                ax.text(x, y, str(nr), fontsize=7, verticalalignment='bottom',
                        horizontalalignment='center', bbox=bbox)

        return ax


if __name__ == '__main__':
    pass

