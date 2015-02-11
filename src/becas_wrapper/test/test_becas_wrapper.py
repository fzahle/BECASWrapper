
import unittest
import numpy as np

from fusedwind.turbine.structure_vt import CrossSectionStructureVT, MaterialProps
from becas_wrapper.cs2dtobecas import CS2DtoBECAS

def make_cs2d():

    cs2d = CrossSectionStructureVT()
    cs2d.airfoil.initialize(np.loadtxt('ffaw3301.dat'))
    # DEFINE MATERIALS
    uniax = MaterialProps()
    uniax.E1 = 40e9
    uniax.E2 = 10e9
    uniax.E3 = 10e9
    uniax.nu12 = 0.28
    uniax.nu13 = 0.28
    uniax.nu23 = 0.4
    uniax.G12 = 4e9
    uniax.G13 = 4e9
    uniax.G23 = 3.571e9
    uniax.rho = 1900
    uniax.materialname = 'uniax'

    biax = MaterialProps()
    biax.E1 = 12e9
    biax.E2 = 12e9
    biax.E3 = 10e9
    biax.nu12 = 0.5
    biax.nu13 = 0.28
    biax.nu23 = 0.28
    biax.G12 = 10e9
    biax.G13 = 3.8e9
    biax.G23 = 3.8e9
    biax.rho = 1890
    biax.materialname = 'biax'

    triax = MaterialProps()
    triax.E1 = 20e9
    triax.E2 = 10e9
    triax.E3 = 10e9
    triax.nu12 = 0.5
    triax.nu13 = 0.28
    triax.nu23 = 0.28
    triax.G12 = 7.5e9
    triax.G13 = 4e9
    triax.G23 = 4e9
    triax.rho = 1860
    triax.materialname = 'triax'

    core = MaterialProps()
    core.E1 = 50e6
    core.E2 = 50e6
    core.E3 = 50e6
    core.nu12 = 0.4
    core.nu13 = 0.4
    core.nu23 = 0.4
    core.G12 = 17.857e6
    core.G13 = 17.857e6
    core.G23 = 17.857e6
    core.rho = 80
    core.materialname = 'core'

    # add materials to airfoil
    cs2d.add_material('uniax', uniax)
    cs2d.add_material('biax', biax)
    cs2d.add_material('triax', triax)
    cs2d.add_material('core', core)

    # --------
    # Region 1: TE
    # --------
    r = cs2d.add_region('REGION01')
    r.s0 = -1.00
    r.s1 = -0.90

    l = r.add_layer('l1')
    l.thickness = .006
    l.angle = 0
    l.materialname = 'TRIAX'
    l.plyname = 'Ply01'

    # --------
    # Region 2: TE sandwich
    # --------
    r = cs2d.add_region('REGION02')
    r.s0 = -0.90
    r.s1 = -0.50

    l = r.add_layer('l1')
    l.thickness = .002
    l.angle = 0
    l.materialname = 'TRIAX'
    l.plyname = 'Ply02'

    l = r.add_layer('l2')
    l.thickness = .015
    l.angle = 0
    l.materialname = 'CORE'
    l.plyname = 'Ply03'

    l = r.add_layer('l3')
    l.thickness = .002
    l.angle = 0
    l.materialname = 'TRIAX'
    l.plyname = 'Ply04'

    # --------
    # Region 3: spar cap
    # --------
    r = cs2d.add_region('REGION03')
    r.s0 = -0.50
    r.s1 = -0.20

    l = r.add_layer('l1')
    l.thickness = .004
    l.angle = 0
    l.materialname = 'UNIAX'
    l.plyname = 'Ply02'

    # --------
    # Region 4: LE edge sandwich
    # --------
    r = cs2d.add_region('REGION04')
    r.s0 = -0.20
    r.s1 = 0.20

    l = r.add_layer('l1')
    l.thickness = .002
    l.angle = 0
    l.materialname = 'TRIAX'
    l.plyname = 'Ply06'

    l = r.add_layer('l2')
    l.thickness = .010
    l.angle = 0
    l.materialname = 'CORE'
    l.plyname = 'Ply07'

    l = r.add_layer('l3')
    l.thickness = .002
    l.angle = 0
    l.materialname = 'TRIAX'
    l.plyname = 'Ply08'

    # -----------
    # Regions 5-8
    # -----------
    r = cs2d.add_region('REGION05')
    cs2d.REGION05 = cs2d.REGION03.copy()
    cs2d.REGION05.s0 = 0.20
    cs2d.REGION05.s1 = 0.50
    r = cs2d.add_region('REGION06')
    cs2d.REGION06 = cs2d.REGION02.copy()
    cs2d.REGION06.s0 = 0.5
    cs2d.REGION06.s1 = 0.9
    r = cs2d.add_region('REGION07')
    cs2d.REGION07 = cs2d.REGION01.copy()
    cs2d.REGION07.s0 = 0.9
    cs2d.REGION07.s1 = 1.0

    # --------
    # Webs
    # --------
    r = cs2d.add_web('WEB00')
    r.s0 = 1.
    r.s1 = -1.
    l = r.add_layer('l1')
    l.thickness = .002
    l.angle = 0
    l.materialname = 'BIAX'
    l.plyname = 'Ply09'

    r = cs2d.add_web('WEB01')
    r.s0 = -.5
    r.s1 = 0.5
    l = r.add_layer('l1')
    l.thickness = .002
    l.angle = 0
    l.materialname = 'BIAX'
    l.plyname = 'Ply09'

    l = r.add_layer('l2')
    l.thickness = .010
    l.angle = 0
    l.materialname = 'CORE'
    l.plyname = 'Ply10'

    l = r.add_layer('l3')
    l.thickness = .002
    l.angle = 0
    l.materialname = 'BIAX'
    l.plyname = 'Ply11'

    r = cs2d.add_web('WEB02')
    cs2d.WEB02 = cs2d.WEB01.copy()
    cs2d.WEB02.s0 = -0.2
    cs2d.WEB02.s1 =  0.2

    # =============================================================================
    ### REMESH AIRFOIL, CREATE BECAS INPUT FILES
    # =============================================================================
    # Number of mesh points along the airfoil. Should be higher as the number of
    # points in the airfoil input file
    b = CS2DtoBECAS()
    b.cs2d = cs2d
    b.total_points = 150
    # dominant regions: spar caps
    b.dominant_elsets = ['REGION03', 'REGION05']
    b.path_shellexpander = '/Users/frza/svn/BECAS/shellexpander/src'
    b.run()
    return b


class BECASWrapperTestCase(unittest.TestCase):

    def setUp(self):
        pass
        
    def tearDown(self):
        pass
        
    # add some tests here...
    
    #def test_BECASWrapper(self):
        #pass
        
if __name__ == "__main__":
    # unittest.main()
    b = make_cs2d()
    
