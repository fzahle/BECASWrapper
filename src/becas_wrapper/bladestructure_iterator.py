
import numpy as np
import logging
import time

from openmdao.main.api import Assembly, Component
from openmdao.lib.drivers.api import CaseIteratorDriver
from openmdao.lib.datatypes.api import Str, VarTree, Float, List, Array

from fusedwind.interface import implement_base
from fusedwind.turbine.structure_vt import BeamStructureVT
from fusedwind.turbine.geometry_vt import BladePlanformVT
from fusedwind.turbine.blade_structure import BeamStructureCSCode, \
                                              StressRecoveryCSCode
from fusedwind.turbine.rotoraero_vt import LoadVectorCaseList

from cs2dtobecas import CS2DtoBECAS
from becas_wrapper import BECASWrapper


class ComputeHAWC2BeamProps(Component):
    """
    Component that postprocesses the individual outputs from the
    BECAS case runs returning a BeamStructureVT, mass and mass moment.
    """

    pf = VarTree(BladePlanformVT(), iotype='in')
    hub_radius = Float(iotype='in')
    cs2d_cases = List(iotype='in')
    g = Float(9.81, iotype='in')
    beam_structure = VarTree(BeamStructureVT(), iotype='out')
    mass = Float(iotype='out')
    mass_moment = Float(iotype='out')

    def execute(self):

        self.beam_structure = self.beam_structure.copy()

        ni = len(self.cs2d_cases)
        
        for name in self.beam_structure.list_vars():
            var = getattr(self.beam_structure, name)
            if name == 'J':
                nname = 'K'
            else:
                nname = name
            if isinstance(var, np.ndarray):
                var = np.zeros(ni)
                for i, h2d in enumerate(self.cs2d_cases):
                    try:
                        var[i] = getattr(h2d, nname)
                    except:
                        pass

            setattr(self.beam_structure, name, var)

        self.beam_structure.s = self.pf.s * self.pf.blade_length
        self.mass = np.trapz(self.beam_structure.dm, self.beam_structure.s)
        self.mass_moment = np.trapz(self.g * self.beam_structure.dm * (self.beam_structure.s + self.hub_radius), 
                                                                      self.beam_structure.s)

        print 'Mass: ', self.mass
        print 'Mass moment: ', self.mass_moment

        self.beam_structure.x_e += (0.5 - self.pf.p_le) * self.pf.chord * self.pf.blade_length
        self.beam_structure.x_cg += (0.5 - self.pf.p_le) * self.pf.chord * self.pf.blade_length
        self.beam_structure.x_sh += (0.5 - self.pf.p_le) * self.pf.chord * self.pf.blade_length


class ProcessFailures(Component):

    failureIn = List(iotype='in')
    max_failure = Array(iotype='out')

    def execute(self):

        nsec = len(self.failureIn)
        ncases = len(self.failureIn[0].cases)
        self.max_failure = np.zeros((ncases, nsec))
        for i, case in enumerate(self.failureIn):
            try:
                self.max_failure[:, i] = np.asarray(self.failureIn[i].cases)
            except:
                self.max_failure[:, i] = np.zeros(ncases)


@implement_base(BeamStructureCSCode)
class BECASBeamStructure(Assembly):
    """
    Assembly that implements the FUSED-Wind I/O interface 
    defined in BeamStructureCSCode.
    The assembly sets up a series of BECAS computations that iterate
    over a list of CrossSectionStructureVT's and computes beam
    structural properties suitable for an aeroelastic solver.
    """

    becas_inputs = Str('becas_inputs', desc='Relative path to BECAS input files', iotype='in')
    section_name = Str('BECAS_SECTION', desc='Section name used by shellexpander and BECASWrapper', iotype='in')

    cs2d = List(iotype='in', desc='Blade cross sectional structure geometry')
    pf = VarTree(BladePlanformVT(), iotype='in', desc='Blade planform with same spanwise discretization as cs2d')

    beam_structure = VarTree(BeamStructureVT(), iotype='out', desc='Structural beam properties')

    def _pre_execute(self):
        super(BECASBeamStructure, self)._pre_execute()

        self.tt = time.time()

    def _post_execute(self):
        super(BECASBeamStructure, self)._post_execute()

        t = time.time() - self.tt
        self._logger.info('BECASBeamStructure time: %f' % t)


@implement_base(StressRecoveryCSCode)
class BECASStressRecovery(Assembly):
    """
    Assembly that implements the FUSED-Wind I/O interface 
    defined in StressRecoveryCSCode.
    The assembly sets up a series of BECAS computations that iterate
    over a list of CSLoadVectorCaseArray's and computes stresses and
    strains. The assembly assumes that BECAS has already been executed
    to compute the stiffness matrix of the cross sections and that
    a series of .mat restart files are present in the base_dir.
    """

    load_cases = List(LoadVectorCaseList, iotype='in', 
                           desc='List of section load vectors used to perform'
                                'failure analysis')

    failure = Array(iotype='out', desc='Failure parameter')

    def _pre_execute(self):
        super(BECASStressRecovery, self)._pre_execute()

        self.tt = time.time()

    def _post_execute(self):
        super(BECASStressRecovery, self)._post_execute()

        t = time.time() - self.tt
        self._logger.info('BECASStressRecovery time: %f' % t)


def becas_configure_stiffness_calc(kls):

    # uncomment to keep Sim-* directories
    # import os
    # os.environ['OPENMDAO_KEEPDIRS'] = '1'

    kls.add('blade_beam_st', BECASBeamStructure())
    kls.driver.workflow.add('blade_beam_st')

    cls = kls.blade_beam_st

    cls.add('cid', CaseIteratorDriver())
    cls.driver.workflow.add('cid')
    cls.create_passthrough('cid.sequential')
    cls.sequential = False

    cls.add('a2b', CS2DtoBECAS())
    cls.cid.workflow.add('a2b')

    cls.create_passthrough('a2b.path_shellexpander')
    cls.connect('becas_inputs', 'a2b.becas_inputs')
    cls.connect('section_name', 'a2b.section_name')

    # add the CrossSectionStructureVT as parameter to the cid
    cls.cid.add_parameter('a2b.cs2d')
    # the parent assembly will connect to this with a list of cases
    # for each radial position
    cls.connect('cs2d', 'cid.case_inputs.a2b.cs2d')

    # add becas
    cls.add('becas', BECASWrapper())
    cls.cid.workflow.add('becas')

    cls.becas.exec_mode = 'octave'
    cls.becas.analysis_mode = 'stiffness'

    # connect path_input constructed by a2b to becas
    cls.connect('a2b.path_input', 'becas.path_input')
    cls.connect('a2b.cs2d.s/pf.blade_length', 'becas.spanpos')

    # path_becas needs to be set at runtime 
    cls.create_passthrough('becas.path_becas')
    # declare outputs
    cls.cid.add_response('becas.hawc2_crossVT')
    cls.create_passthrough('cid.case_outputs.becas.hawc2_crossVT')

    # postprocess each case to generate the BeamStructureVT output
    # moved to parent assembly to avoid pickling error
    cls.add('postpro', ComputeHAWC2BeamProps())
    cls.driver.workflow.add('postpro')
    cls.connect('pf', 'postpro.pf')
    cls.connect('cid.case_outputs.becas.hawc2_crossVT', 'postpro.cs2d_cases')
    cls.connect('postpro.beam_structure', 'beam_structure')
    cls.create_passthrough('postpro.mass')
    cls.create_passthrough('postpro.mass_moment')
    cls.create_passthrough('postpro.hub_radius')

    # uncomment to get full debug output
    cls.log_level = logging.DEBUG
    cls.a2b.log_level = logging.DEBUG
    cls.becas.log_level = logging.DEBUG


def becas_configure_stress_recovery(kls):

    # uncomment to keep Sim-* directories
    # import os
    # os.environ['OPENMDAO_KEEPDIRS'] = '1'

    kls.add('blade_strength', BECASStressRecovery())
    kls.driver.workflow.add('blade_strength')

    cls = kls.blade_strength

    cls.add('cid', CaseIteratorDriver())
    cls.driver.workflow.add('cid')
    cls.create_passthrough('cid.sequential')
    cls.sequential = False

    # add becas
    cls.add('becas', BECASWrapper())
    cls.cid.workflow.add('becas')

    cls.becas.exec_mode = 'octave'
    cls.becas.analysis_mode = 'stress_recovery'
    cls.create_passthrough('becas.path_becas')

    # add parameters and responses
    cls.cid.add_parameter('becas.load_cases')
    cls.cid.add_response('becas.max_failure')
    cls.cid.add_response('becas.max_failure_ks')
    cls.connect('load_cases', 'cid.case_inputs.becas.load_cases')
    # cls.create_passthrough('cid.case_inputs.becas.load_cases')

    # cls.cid.add_response('becas.stress')
    # cls.cid.add_response('becas.strain')

    # add postprocessing components
    cls.add('process_failure', ProcessFailures())
    cls.add('process_failure_ks', ProcessFailures())
    cls.driver.workflow.add(['process_failure', 'process_failure_ks'])
    cls.connect('cid.case_outputs.becas.max_failure', 'process_failure.failureIn')
    cls.connect('cid.case_outputs.becas.max_failure_ks', 'process_failure_ks.failureIn')
    cls.connect('process_failure.max_failure', 'failure')
    cls.create_passthrough('process_failure_ks.max_failure', alias='failure_ks')

    cls.log_level = logging.DEBUG
    cls.becas.log_level = logging.DEBUG
