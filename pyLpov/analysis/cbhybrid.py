from pyLpov.analysis.calibration import Calibration
from pyLpov.io.dataset import DataSet
#from pyLpov.analysis.cberp import CBERP
#from pyLpov.analysis.cbssvep import CBSSSVEP
from pyLpov.machine_learning.evaluate import evaluate_pipeline


class CBHYBRID(object):

    def __init__(self, datapath=None, experiment=None, pipeline=None, 
                 outdir=None, analysis=[],fitted=False
                 ):
        self.datapath = datapath
        self.experiment = experiment
        self.pipelines = pipeline
        self.outdir = outdir      
        self.paradigm = Calibration.parse_paradigm(self.experiment)
        self.n_prdg = 0
        self.analysis = analysis
        self.fitted = False   

    def run_analysis(self):
        cnt, data_sets = DataSet.convert_raw(self.datapath, self.paradigm)
        for ds in data_sets:
            self.single_analysis(cnt, ds)


    def single_analysis(self, cnt, dataset): 
        filter_opts = self.pipelines[dataset.paradigm.paradigmType+'_filter']
        pipeline_opts = self.pipelines[dataset.paradigm.paradigmType+'_pipeline']
        dataset.get_epochs(cnt, filter_opts)
        anls = Calibration(datapath=self.datapath, paradigm=dataset.paradigm, 
                               pipeline={'pipeline' : pipeline_opts}, outdir=self.outdir
                               )
        anls.run_analysis(dataset)
        self.analysis.append(anls)


    def save(self):
        for anls in self.analysis:
            anls.save()