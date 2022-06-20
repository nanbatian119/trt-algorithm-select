import torch
import numpy as np
import os
import onnxruntime
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnx_graphsurgeon as gs
import onnx
import ctypes
import copy
ctypes.cdll.LoadLibrary('./layernorm/LayerNorm.so')
G_LOGGER = trt.Logger(trt.Logger.ERROR)
class MyAlgorithmSelector(trt.IAlgorithmSelector):

    def __init__(self, keepAll=True):
        super(MyAlgorithmSelector, self).__init__()
        self.keepAll = keepAll

    def select_algorithms(self, layerAlgorithmContext, layerAlgorithmList):
        print(layerAlgorithmContext.name)
        print(len(layerAlgorithmList))
        result = list((range(len(layerAlgorithmList))))
        #result = []
        if layerAlgorithmContext.name == 'MatMul_115':
            for index, algorithm  in enumerate(layerAlgorithmList):
                print(algorithm.algorithm_variant.tactic)
            result = [1]
            #result = [ index for index,algorithm in enumerate(layerAlgorithmList) if algorithm.algorithm_variant.implementation == 0x1fc87d7eb370bb7a ]
        return result

    def report_algorithms(self, modelAlgorithmContext, modelAlgorithmList):
        
        for i in range(len(modelAlgorithmContext)):
            context = modelAlgorithmContext[i]
            algorithm = modelAlgorithmList[i]

            print("Layer%4d:%s" % (i, context.name))
            nInput = context.num_inputs
            nOutput = context.num_outputs
            for j in range(nInput):
                ioInfo = algorithm.get_algorithm_io_info(j)
                print("    Input [%2d]:%s,%s,%s,%s" % (j, context.get_shape(j), ioInfo.dtype, ioInfo.strides, ioInfo.tensor_format))
            for j in range(nOutput):
                ioInfo = algorithm.get_algorithm_io_info(j + nInput)
                print("    Output[%2d]:%s,%s,%s,%s" % (j, context.get_shape(j + nInput), ioInfo.dtype, ioInfo.strides, ioInfo.tensor_format))
            print("    algorithm:[implementation:%d,tactic:%d,timing:%fms,workspace:%dMB]"% \
                  (algorithm.algorithm_variant.implementation,
                   algorithm.algorithm_variant.tactic,
                   algorithm.timing_msec,
                   algorithm.workspace_size))
class Test(object):
       
    def infer_algorithm_selector_onnx(self, onnx_file, dynamic_list):
        builder = trt.Builder(G_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.max_workspace_size = 3 << 30
        parser = trt.OnnxParser(network, G_LOGGER)
        with open(onnx_file, 'rb') as model:
            if not parser.parse(model.read()):
                print("Failed parsing onnx file!")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                exit()
        print("Succeeded parsing onnx file!")
        
        input_ids = network.get_input(0)
        bbox = network.get_input(1)
        images = network.get_input(2)
        input_num = network.num_inputs
        for i in range(input_num):
            input_i = network.get_input(i)
            shape = list(input_i.shape)
            min_shape = copy.deepcopy(shape)
            min_shape[0] = dynamic_list[0]
            opt_shape = copy.deepcopy(shape)
            opt_shape[0] = dynamic_list[1]
            max_shape = copy.deepcopy(shape)
            max_shape[0] = dynamic_list[2]
            print(input_i.name, min_shape, opt_shape, max_shape)
            profile.set_shape(input_i.name, min_shape, opt_shape, max_shape)

        config.add_optimization_profile(profile)
        config.algorithm_selector = MyAlgorithmSelector(True)  # set algorithm_selector
        
        engineString = builder.build_serialized_network(network, config)
        with open('./err.plan', 'wb') as f:
            f.write(engineString)


    def t(self):
        print('bad, start...\n')
        # bad
        self.infer_algorithm_selector_onnx('./onnx/test.onnx', [1,6,12])
        
        print('good, start...\n')
        # good
        self.infer_algorithm_selector_onnx('./onnx/test.onnx', [12,12,12])

if __name__ == '__main__':
    test = Test()
    test.t()

