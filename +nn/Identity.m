classdef Identity < nn.Module
    properties
    end
    methods
        function obj = Identity()
            obj = obj@nn.Module();
        end
        function output = fprop(obj, input)
            obj.output = input;
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            obj.grad_input = grad_output;
            grad_input = obj.grad_input;
        end
    end
end