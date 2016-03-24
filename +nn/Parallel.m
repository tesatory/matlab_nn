classdef Parallel < nn.Contrainer
    properties
    end
    methods
        function obj = Parallel()
            obj = obj@nn.Contrainer();
        end
        function output = fprop(obj, input)
            obj.output = {};
            for i = 1:length(obj.modules)
                obj.output{i} = obj.modules{i}.fprop(input{i});
            end
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            obj.grad_input = {};
            for i = 1:length(obj.modules)
                obj.grad_input{i} = obj.modules{i}.bprop(input{i}, grad_output{i});
            end
            grad_input = obj.grad_input;
        end
    end
end