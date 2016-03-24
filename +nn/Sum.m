classdef Sum < nn.Module
    properties
        dim
    end
    methods
        function obj = Sum(dim)
            obj = obj@nn.Module();
            obj.dim = dim;
        end
        function output = fprop(obj, input)
            obj.output = squeeze(sum(input, obj.dim));
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            sz = size(input);
            sz(obj.dim) = 1;
            grad_output = reshape(grad_output, sz);
            sz(:) = 1;
            sz(obj.dim) = size(input, obj.dim);
            obj.grad_input = repmat(grad_output, sz);
            grad_input = obj.grad_input;
        end
    end
end