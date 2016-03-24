classdef Loss < handle
    properties
    end
    methods
        function obj = Loss()
            obj = obj@handle();
        end
        function cost = fprop(obj, input, target)
            assert(false, obj.name)
        end
        function grad_input = bprop(obj, input, target)
            assert(false, obj.name)
        end
    end
end