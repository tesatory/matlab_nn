classdef ElemMult < nn.Module
    properties
        weight
    end
    methods
        function obj = ElemMult(w)
            obj = obj@nn.Module();
            obj.weight = w;
        end
        function output = fprop(obj, input)
            obj.output = bsxfun(@times, input, obj.weight);
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            obj.grad_input = bsxfun(@times, grad_output, obj.weight);
            grad_input = obj.grad_input;
        end
    end
end