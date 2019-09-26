from pyLpov.paradigms import base 
from pyLpov.paradigms import erp, ssvep


class HYBRID(object):

    def __init__(self, title='LARESI_HYBRID', paradigmType=2, experimentMode=0,
                 ERPparadigm=None, SSVEPparadigm=None
                 ):
        
        self.title = title
        self.paradigmType = base.PARADIGM_TYPE[paradigmType]
        self.experimentMode = base.OPERATION_MODE[experimentMode]
        self.ERP = ERPparadigm
        self.SSVEP = SSVEPparadigm

    def from_json(self, json_data):
        self.paradigmType = base.PARADIGM_TYPE[json_data['paradigmType']]
        self.experimentMode = base.OPERATION_MODE[json_data['experimentMode']]        
        self.ERP = erp.ERP.from_json(self.parse_json('ERP', json_data))
        self.SSVEP = ssvep.SSVEP.from_json(self.parse_json('SSVEP', json_data))
        return self

    def parse_json(self, substr, json_data):
        paradigm_dict = {}
        
        for key, value in json_data.iteritems():   # iter on both keys and values
            if key.startswith(substr):
                paradigm_dict[key.replace(substr+'_','')] = value
        
        paradigm_dict['experimentMode'] = json_data['experimentMode']

        return paradigm_dict