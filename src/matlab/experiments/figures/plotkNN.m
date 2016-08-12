function plotkNN()

% read in the data

% for ik = [1, 5:5:100]
%     save_str = ['H:\Data\saved_data\adjacency\IndianPines_', sprintf('k%d', ik)];
%     load(char(save_str))
%     kMatrix = varString;
%     clear varString
%     
%     % figure
%     f = figure;
%     spy(kMatrix, '-k', 1e-20);
%     save_path = ['E:\cloud_drives\dropbox\Apps\', ...
%                     '\ShareLaTeX\thesis - masters\tex\figures\ch6\exp1\'];
%     save_str = char([save_path, ...
%         sprintf('indianpines_kmat_%d', ik)]);
%     print(save_str, '-depsc');
%     close(gcf);
% 
% 
% end
% 
% 
% for ik = [1, 5:5:40]
%     save_str = ['H:\Data\saved_data\adjacency\Pavia_', sprintf('k%d', ik)];
%     load(char(save_str))
%     kMatrix = varString;
%     clear varString
%     
%     % figure
%     f = figure;
%     spy(kMatrix, '-k', 1e-20);
%     save_path = ['E:\cloud_drives\dropbox\Apps\', ...
%                     '\ShareLaTeX\thesis - masters\tex\figures\ch6\exp1\'];
%     save_str = char([save_path, ...
%         sprintf('pavia_kmat_%d', ik)]);
%     print(save_str, '-depsc');
%     close(gcf);
% 
% 
% end

for ik = 1:10
    save_str = ['H:\Data\saved_data\sigma_results\IndianPines_k20_sigma', sprintf('%d', ik)];
    load(char(save_str))
    kMatrix = varString;
    clear varString
    
    % figure
    f = figure;
    spy(kMatrix, '-k', 1e-20);
    save_path = ['E:\cloud_drives\dropbox\Apps\', ...
                    '\ShareLaTeX\thesis - masters\tex\figures\ch6\exp1\'];
    save_str = char([save_path, ...
        sprintf('indianpines_sigmat_%d', ik)]);
    print(save_str, '-depsc');
    close(gcf);


end

for ik = 1:10
    save_str = ['H:\Data\saved_data\sigma_results\Pavia_sigma20_sigma', sprintf('%d', ik)];
    load(char(save_str))
    kMatrix = varString;
    clear varString
    
    % figure
    f = figure;
    spy(kMatrix, '-k', 1e-20);
    save_path = ['E:\cloud_drives\dropbox\Apps\', ...
                    '\ShareLaTeX\thesis - masters\tex\figures\ch6\exp1\'];
    save_str = char([save_path, ...
        sprintf('pavia_sigmat_%d', ik)]);
    print(save_str, '-depsc');
    close(gcf);


end


end