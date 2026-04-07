from spotify_confidence.analysis.constants import BOOTSTRAP, CHI2, TTEST, ZTEST, ZTESTLINREG
from spotify_confidence.analysis.frequentist.confidence_computers.bootstrap_computer import BootstrapComputer
from spotify_confidence.analysis.frequentist.confidence_computers.chi_squared_computer import ChiSquaredComputer
from spotify_confidence.analysis.frequentist.confidence_computers.t_test_computer import TTestComputer
from spotify_confidence.analysis.frequentist.confidence_computers.z_test_computer import ZTestComputer
from spotify_confidence.analysis.frequentist.confidence_computers.z_test_linreg_computer import ZTestLinregComputer

confidence_computers = {
    CHI2: ChiSquaredComputer(),
    TTEST: TTestComputer(),
    ZTEST: ZTestComputer(),
    BOOTSTRAP: BootstrapComputer(),
    ZTESTLINREG: ZTestLinregComputer(),
}
