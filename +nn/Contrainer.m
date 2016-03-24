classdef Contrainer < nn.Module
    properties
        modules = {};
    end
    methods
        function obj = Contrainer()
            obj = obj@nn.Module();
        end
        function add(obj, m)
            obj.modules{end+1} = m;
        end
        function update(obj, params)            
            for i = 1:length(obj.modules)
                obj.modules{i}.update(params);
            end
        end        
        function share(obj, m)
            for i = 1:length(obj.modules)
                obj.modules{i}.share(m.modules{i});
            end
        end
    end
end