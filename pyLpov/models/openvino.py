
class OpenVinoModel:
    def __init__(self, net):
        self.net = net
        self.input_name = next(iter(self.net.input_info))
        self.output_name = next(iter(self.net.outputs))
        self.request = self.net.requests[0]
        
        
    def predict(self, epoch):
        self.request.infer({self.input_name: epoch})

        # read the result
        prediction_openvino_blob = self.request.output_blobs[self.output_name]
        if prediction_openvino_blob.buffer.size == 1:
            return prediction_openvino_blob.buffer.item()
        else:
            return prediction_openvino_blob.buffer
        # return prediction_openvino_blob.buffer.item()       
        