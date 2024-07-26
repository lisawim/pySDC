import pytest
import warnings


@pytest.mark.base
def test_problematic_main():
    from pySDC.projects.DAE import mainFullyImplicitDAE

    mainFullyImplicitDAE()


@pytest.mark.base
def test_synch_gen_playground_main():
    from pySDC.projects.DAE import mainSyncMachine

    warnings.filterwarnings('ignore')
    mainSyncMachine()
    warnings.resetwarnings()
