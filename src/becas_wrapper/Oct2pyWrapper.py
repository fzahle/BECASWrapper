# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:21:52 2013

@author: dave
"""

from __future__ import division, print_function, absolute_import
#import os
import numpy as np
from oct2py import octave
import unittest
import six

## ===========================================================================
#PY2 = sys.version[0] == '2'
#PY3 = sys.version[0] == '3'
#if PY2:
#    # Python 2
#    import codecs
#    def u(obj):
#        """Make unicode object"""
#        return codecs.unicode_escape_decode(obj)[0]
#else:
#    # Python 3
#    def u(obj):
#        """Return string as it is"""
#        return obj

u = six.text_type

# create classes to store all the nested Octave variables

class NestedOctave:
    """
    Create a nested variable tree as in Octave
    """
    def __init__(self, octave, parent, debug=False):
        """
        
        OBSOLETE: version 1.0 of oct2py gets this right!!
        
        oct2py doesn't handle nested Octave variables very well. However, the 
        result is an empty NumPy array for which the dtypes are the nested 
        variable names in Octave. This ugly monkey patch gets all the second
        nested variables.
        
        Parameters
        ----------
        
        parent : string
            Octave variable name of type struct (aka as a nested variable)
        
        octave : oct2py._oct2py.Oct2Py object
            A valid living octave instance of oct2py for which all relevant
            BECAS commands have been executed. It should hence contain
            all the relevant BECAS output variables.
        """
        
        # oct2py doesn't handle nested Octave variables very well. However, 
        # the result is a NumPy array for which the dtypes are the nested 
        # variable names in Octave. This ugly monkey patch gets all the second
        # nested variables.
        oct_var = octave.call(parent)
        for child in oct_var.dtype.names:
            # there seems to be a bug (or is it a feature?) in oct2py:
            # long nested variable names are truncated, or when ending with r?
            # Ugly monkey patch follows to get back the missing r at the end
            if child == 'AlphaPrincipleAxis_ElasticCente':
                child += 'r'
            
            # can't use eval because that is only for expressions (no =). 
            # Exec can also supports statements
            expr = "self.%s = octave.call('%s.%s')" % (child, parent, child)
            
            if debug:
                print(expr)
            
            exec compile(expr, "<string>", "exec")
        
        # and set the class name to the parent name, but that needs to be
        # a string here....
        self.__class__.__name__ = parent


class OctavePythonRoundtripTests(unittest.TestCase):
    
    
    def test_becas_input_methods_utils(self):
        """
        Verify that reading the input files with Python and Octave/BECAS
        gives exactly the same result. This is a test to verify that the 
        mechanism of putting and getting arrays from Octave works exactly
        the same as loading the files in Octave.
        """
        # make sure we have a fresh new and clean octave session
        octave.restart()
        
        p_becas = '/home/dave/Repositories/DTU/BECAS/src/matlab'
        exampl = 'BECAS_examples/WTAirfoil'
        
        # load the input files with Python
        nl_2d = np.loadtxt('%s/%s/N2D.in' % (p_becas, exampl))
        el_2d = np.loadtxt('%s/%s/E2D.in' % (p_becas, exampl))
        emat = np.loadtxt('%s/%s/EMAT.in' % (p_becas, exampl))
        matprops = np.loadtxt('%s/%s/MATPROPS.in' % (p_becas, exampl))
        
        # put them into Octave
        octave.put('nl_2d', nl_2d)
        octave.put('el_2d', el_2d)
        octave.put('emat', emat)
        octave.put('matprops', matprops)
        
        # short hand notation
        oc = octave.run
        oc("cd('%s')" % p_becas)
        oc('BECAS_SetupPath')
        p1, p2 = exampl.split('/')
        oc("options.foldername=fullfile('%s','%s');" % (p1, p2))
        oc("[ utils ]=BECAS_Utils(options);")
        
        # compare utils and python input from the octave side
        oc("c1 = (nl_2d == utils.nl_2d);")
        oc("c2 = (el_2d == utils.el_2d);")
        oc("c3 = (emat == utils.emat);")
        oc("c4 = (matprops == utils.matprops);")
        
        # load the comparisons into Python
        c1 = octave.get('c1')
        c2 = octave.get('c2')
        c3 = octave.get('c3')
        c4 = octave.get('c4')
        
        # and make sure they all evaluate to 1
        self.assertTrue( c1.min() == 1 )
        self.assertTrue( c2.min() == 1 )
        self.assertTrue( c3.min() == 1 )
        self.assertTrue( c4.min() == 1 )
        
        # and see if the passing of the input arguments works properly
        oc("[ utils ]=BECAS_Utils(options, nl_2d, el_2d, emat, matprops);")
        oc("c5 = (nl_2d == utils.nl_2d);")
        oc("c6 = (el_2d == utils.el_2d);")
        oc("c7 = (emat == utils.emat);")
        oc("c8 = (matprops == utils.matprops);")
        
        # load the comparisons into Python
        c5 = octave.get('c5')
        c6 = octave.get('c6')
        c7 = octave.get('c7')
        c8 = octave.get('c8')
        
        # and make sure they all evaluate to 1
        self.assertTrue(c5.min() == 1)
        self.assertTrue(c6.min() == 1)
        self.assertTrue(c7.min() == 1)
        self.assertTrue(c8.min() == 1)    
    
def runbecas():

    # =========================================================================
    # add all the BECAS source files to the Octave path
    p_becas = '/home/dave/Repositories/DTU/BECAS/src/matlab/'
    
    # make sure we have a fresh new and clean octave session
    octave.restart()
    
    # short hand notation
    oc = octave.run
    
    # set current working directory of the Python console to BECAS main folder
    #os.chdir(path_becas)
    # and the same for Octave
    oc("cd('%s')" % p_becas)
    
    ## just to be sure, add the OpenMDAO becas wrapper also to the Octave path
    #cwd = u'/home/dave/Repositories/DTU-Gitlab-Redmine/becas_wrapper/trunk/'
    #cwd += u'becas_wrapper/src/becas_wrapper'
    #octave.addpath(u'%s' % cwd)
    
    #for k in octave.addpath("").split(':'): print(k)
    
    #octave.call('runBECAS.m')
    # =========================================================================
    # do everything that is done in runBECAS.m
    oc('tic')
    oc('BECAS_SetupPath')
    #dummy = octave.BECAS_SetupPath(1)
    oc("options.foldername=fullfile('BECAS_examples','WTAirfoil');")
    ## nested variables (struct datatypes) are not returned properly by oct2py
    #options = oc('options')
    
    # Build arrays for BECAS
    #oc("[ utils ]=BECAS_Utils(options, nl_2d, el_2d, emat, matprops);")
    oc("[ utils ]=BECAS_Utils(options);")
    
    # Check mesh quality
    oc("[ meshcheck ] = BECAS_CheckMesh( utils );")
    
    # Call BECAS for the evaluation of the cross section stiffness matrix
    oc("[ constitutive.Ks, solutions ] = BECAS_Constitutive_Ks(utils);")
    
    # Call BECAS module for the evaluation of the cross section mass matrix
    oc("[ constitutive.Ms ] = BECAS_Constitutive_Ms(utils);")
    
    # Call BECAS module for the evaluation of the cross section properties
    oc("[ csprops ] = BECAS_CrossSectionProps(constitutive.Ks, utils);")
    
    # Recover strains and stress for a given force and moment vector
    # Load vector
    oc("theta0=[0 0 0 0 0 1];")
    
    # Calculate strains
    oc("[ strain ] = BECAS_RecoverStrains(theta0, solutions, utils);")
    
    # Calculate stresses
    oc("[ stress ] = BECAS_RecoverStresses( strain, utils );")
    
    # Output of results to HAWC2 st file
    oc("RadPos=1") # Define radial position
    oc("[hawc2] = BECAS_Becas2Hawc2(false,RadPos,constitutive,csprops,utils);")
    
    # Check failure criteria if input file is given only
    #oc('[fail]=BECAS_CheckFailure(utils,stress.MaterialGauss,strain.MaterialGauss)')
    
    # now the nested variable becomes an dictionary holding the different vars
    utils        = octave.get('utils')
    csprops      = octave.get('csprops')
    constitutive = octave.get('constitutive')
    meshcheck    = octave.get('meshcheck')
    options      = octave.get('options')
    solutions    = octave.get('solutions')
    strain       = octave.get('strain')
    stress       = octave.get('stress')
    
    ## save some to mat files
    #oc("save('utils.mat', 'utils')")
    #oc("save('constitutive.mat', 'constitutive')")
    
    #oct_utils = octave.call('utils')
    #for var in oct_utils.dtype.names:
    #    print(u"%s=octave.call('utils.%s')" % (var, var))
    #    # can't use eval because that is only for expressions (no =). Exec can
    #    # also be for statements
    #    code = compile("%s=octave.call('utils.%s')" % (var,var),"<string>","exec")
    #    exec code
    #    getattr(octave.call('utils.%s' % var), var)
    
    print(oc('toc'))
    
    # verify if load() gets what we want it to read
    qq='/home/dave/Repositories/DTU/BECAS/src/matlab/'
    qq+='BECAS_examples/WTAirfoil/'
    
    oc("nl_2d = load('%sN2D.in');" % qq)
    oc("matprops = load('%sMATPROPS.in');" % qq)
    nl_2d = octave.get('nl_2d')
    matprops = octave.get('matprops')
    
    matprops2 = np.loadtxt('%sMATPROPS.in' % qq)
    np.savetxt('%sMATPROPS2.in' % qq, matprops2, fmt='%.06e', delimiter='    ')
    oc("matprops2 = load('%sMATPROPS2.in');" % qq)
    matprops3 = octave.get('matprops2')
    
    ## the roundtrip example from the docs
    #octave.restart()
    #out = octave.test_dataypes()
    #import pprint
    #pprint.pprint(out)

if __name__ == '__main__':
    
    unittest.main()


