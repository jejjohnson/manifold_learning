function bestparameterresults(Parameters)

switch lower(Parameters.exp)
    
    case 'learning'
        %{ 
        Data structure Parameters
        fields: k, sigma, dims, time
        %}
        
        % get parameters
        k = Parameters.k;
        sigma = Parameters.sigma;
        dims = Parameters.dims;
        time = Parameters.time;
        
        % Display header
        fprintf('Algorithm \t k-NN \t \\sigma \t \Dimensions\n');
        
        % Display Result
        fprintf('LE \t %d \t %2.3f \t %d\n', k.le, sigma.le, dims.le, time.le);
        fprintf('SE \t %d \t %2.3f \t %d\n', k.se, sigma.se, dims.se, time.se);
        fprintf('LPP \t %d \t %2.3f \t %d\n', k.lpp, sigma.lpp, dims.lpp, time.lpp);
        fprintf('SEP \t %d \t %2.3f \t %d\n', k.sep, sigma.sep, dims.sep, time.sep);
        
    case 'alignment'
        
%{ 
        Data structure Parameters
        fields: k, sigma, dims, time
        %}
        
        % get parameters
        k = Parameters.k;
        sigma = Parameters.sigma;
        dims = Parameters.dims;
        time = Parameters.time;
        mu = Parameters.mu;
        alpha = Parameters.alpha;
        
        % Display header
        fprintf('Algorithm \t k-NN \t\t \\sigma \t\t \Dimensions \t\n');
        
        % Display Result
        fprintf('Wang \t %d \t %d\t %2.3f \t %2.3f  \t %d \t %d\n', ...
            k.wang{1}, k.wang{2}, ...
            sigma.wang{1}, sigma.wang{2}, ...
            dims.wang{1}, dims.wang{2}, ...
            time.wang{1}, time.wang{2});
        fprintf('SSMA \t %d \t %d\t %2.3f \t %2.3f  \t %d \t %d\n', ...
            k.ssma{1}, k.ssma{2}, ...
            sigma.ssma{1}, sigma.ssma{2}, ...
            dims.ssma{1}, dims.ssma{2}, ...
            time.ssma{1}, time.ssma{2});
        fprintf('SEMA \t %d \t %d\t %2.3f \t %2.3f  \t %d \t %d\n', ...
                    k.sema{1}, k.sema{2}, ...
                    sigma.sema{1}, sigma.sema{2}, ...
                    dims.sema{1}, dims.sema{2}, ...
                    time.sema{1}, time.sema{2});
        
    otherwise
        error('Unrecognized experiment.');

end

end
