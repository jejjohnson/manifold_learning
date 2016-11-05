function [varargout] = plotclassmaps(predImage, originalImage, gtImage, ...
    Options)

if ~isfield(Options, 'imgColors')
    Options.imgColors = [29 15 12];
end


switch lower(Options.type)
    
    case 'initial'
        
        % Display Original Image
        imgColor = originalImage(:,:, Options.imgColors);
        iCMin = min(imgColor(:));
        iCMax = max(imgColor(:));
        imgColor = uint8(255*(imgColor -iCMin)./(iCMax - iCMin));
        imgColorPad = padarray(imgColor, [1 1], 0);
        
        hOriginalImage = figure;
        switch lower(Options.hsi)
            case 'indianpines'
                print('E:\cloud_drives\dropbox\Apps\ShareLaTeX\thesis - masters\tex\figures\indianpines', '-depsc2');
            case 'pavia'
                print('E:\cloud_drives\dropbox\Apps\ShareLaTeX\thesis - masters\tex\figures\pavia', '-depsc2');
            otherwise
                error('Unrecognized hsi image');
        end
        himg = imshow(imgColorPad);
        
        % Display Ground Truth Image
        gtMask = gtImage > 0;
        nClasses = numel(unique(gtImage));
        gtClasses = uint8(255* ind2rgb(gtImage, hsv(nClasses))) .* ...
            repmat(uint8(gtMask), [1 1 3]) + ...
            255 * repmat(uint8(~gtMask), [1 1 3]);
        gtClassesPad = padarray(gtClasses, [1 1 0]);
        
        hgtImage = figure;
        hgt = imshow(gtClassesPad);
        switch lower(Options.hsi)
            case 'indianpines'
                print('E:\cloud_drives\dropbox\Apps\ShareLaTeX\thesis - masters\tex\figures\indianpinesgt', '-depsc2');
            case 'pavia'
                print('E:\cloud_drives\dropbox\Apps\ShareLaTeX\thesis - masters\tex\figures\paviagt', '-depsc2');
            otherwise
                error('Unrecognized hsi image');
        end    
    case 'learning'
        
        % Display Original Image
        imgColor = originalImage(:,:, Options.imgColors);
        iCMin = min(imgColor(:));
        iCMax = max(imgColor(:));
        imgColor = uint8(255*(imgColor -iCMin)./(iCMax - iCMin));
        imgColorPad = padarray(imgColor, [1 1], 0);
        
        hOriginalImage = figure;
        
        himg = imshow(imgColorPad);
        
        switch lower(Options.hsi)
            case 'indianpines'
                print('E:\cloud_drives\dropbox\Apps\ShareLaTeX\thesis - masters\tex\figures\ch5\indianpines', '-depsc2');
            case 'pavia'
                print('E:\cloud_drives\dropbox\Apps\ShareLaTeX\thesis - masters\tex\figures\ch5\pavia', '-depsc2');
            otherwise
                error('Unrecognized hsi image');
        end        
        
        close(gcf);
        
        % Display Ground Truth Image
        gtMask = gtImage > 0;
        nClasses = numel(unique(gtImage));
        gtClasses = uint8(255* ind2rgb(gtImage, hsv(nClasses))) .* ...
            repmat(uint8(gtMask), [1 1 3]) + ...
            255 * repmat(uint8(~gtMask), [1 1 3]);
        gtClassesPad = padarray(gtClasses, [1 1 0]);
        
        hgtImage = figure;
        hgt = imshow(gtClassesPad);
        
        switch lower(Options.hsi)
            case 'indianpines'
                print('E:\cloud_drives\dropbox\Apps\ShareLaTeX\thesis - masters\tex\figures\ch5\indianpinesgt', '-depsc2');
            case 'pavia'
                print('E:\cloud_drives\dropbox\Apps\ShareLaTeX\thesis - masters\tex\figures\ch5\paviagt', '-depsc2');
            otherwise
                error('Unrecognized hsi image');
        end    
        
        alphaData = padarray(double(gtMask)*(3/4) + 1/4, [1 1], 1);
        

        
        close(gcf);
        % Display Resulting Class Labels
        imgClasses = uint8(255*ind2rgb(predImage, hsv(nClasses)));
        imgClassesPad = padarray(imgClasses, [1 1], 0);
        
        hpredImage = figure;
        hpred = imshow(imgClassesPad);
        
        switch lower(Options.hsi)
            case 'indianpines'
                save_path = ['E:\cloud_drives\dropbox\Apps\', ...
                    '\ShareLaTeX\thesis - masters\tex\figures\ch6\exp1\'];
                save_str = char([save_path, ...
                    sprintf('%s_indianpines_classmap', Options.algo)]);
                print(save_str, '-depsc');
            case 'pavia'
                save_path = ['E:\cloud_drives\dropbox\Apps\', ...
                    '\ShareLaTeX\thesis - masters\tex\figures\ch6\exp1\'];
                save_str = char([save_path, ...
                    sprintf('%s_pavia_classmap', Options.algo)]);
                print(save_str, '-depsc');
            otherwise
                error('Unrecognized hsi image');
        end
        set(hpred, 'alphaData', alphaData);
        
        switch lower(Options.hsi)
            case 'indianpines'
                save_path = ['E:\cloud_drives\dropbox\Apps\', ...
                    '\ShareLaTeX\thesis - masters\tex\figures\ch6\exp1\'];
                save_str = char([save_path, ...
                    sprintf('%s_indianpines_classmap_overlay', Options.algo)]);
                print(save_str, '-depsc2');
                    
            case 'pavia'
                save_path = ['E:\cloud_drives\dropbox\Apps\', ...
                    '\ShareLaTeX\thesis - masters\tex\figures\ch6\exp1\'];
                save_str = char([save_path, ...
                    sprintf('%s_pavia_classmap_overlay', Options.algo)]);
                print(save_str, '-depsc2');
            otherwise
                error('Unrecognized hsi image');
        end
        
        close(gcf);
    
    case 'alignment'
        % Display Original Image
        imgColor = originalImage{Options.domain}.img(:,:, [26 15 12]);
        iCMin = min(imgColor(:));
        iCMax = max(imgColor(:));
        imgColor = uint8(255*(imgColor -iCMin)./(iCMax - iCMin));
        imgColorPad = padarray(imgColor, [1 1], 0);
        
        hOriginalImage = figure;
        
        himg = imshow(imgColorPad);
        
        switch lower(Options.domain)
            case 1
                print('E:\cloud_drives\dropbox\Apps\ShareLaTeX\thesis - masters\tex\figures\vcu1', '-depsc2');
            case 2
                print('E:\cloud_drives\dropbox\Apps\ShareLaTeX\thesis - masters\tex\figures\vcu2', '-depsc2');
            otherwise
                error('Unrecognized hsi image');
        end
        close(gcf);
        
        % Display Ground Truth Image
        gtMask = gtImage{Options.domain}.gt > 0;
        nClasses = numel(unique(gtMask>0));
        gtClasses = uint8(255* ind2rgb(gtImage{Options.domain}.gt, hsv(nClasses))) .* ...
            repmat(uint8(gtMask), [1 1 3]) + ...
            255 * repmat(uint8(~gtMask), [1 1 3]);
        gtClassesPad = padarray(gtClasses, [1 1 0]);
        
        hgtImage = figure;
        hgt = imshow(gtClassesPad);
        
        alphaData = padarray(double(gtMask)*(3/4) + 1/4, [1 1], 1);
        switch lower(Options.domain)
            case 1
                print('E:\cloud_drives\dropbox\Apps\ShareLaTeX\thesis - masters\tex\figures\vcugt1', '-depsc2');
            case 2
                print('E:\cloud_drives\dropbox\Apps\ShareLaTeX\thesis - masters\tex\figures\vcugt2', '-depsc2');
            otherwise
                error('Unrecognized hsi image');
        end        
        close(gcf);
        
%         % Display Resulting Class Labels
%         imgClasses = uint8(255*ind2rgb(predImage{Options.domain}, hsv(nClasses)));
%         imgClassesPad = padarray(imgClasses, [1 1], 0);
%         
%         hpredImage = figure;
%         hpred = imshow(imgClassesPad);
%         set(hpred, 'alphaData', alphaData);
%         switch lower(Options.domain)
%             case 1
%                 print('E:\cloud_drives\dropbox\Apps\ShareLaTeX\thesis - masters\tex\figures\vcuclassmap1', '-depsc2');
%             case 2
%                 print('E:\cloud_drives\dropbox\Apps\ShareLaTeX\thesis - masters\tex\figures\vcuclassmap2', '-depsc2');
%             otherwise
%                 error('Unrecognized hsi image');
%         end
%         close(gcf);
    otherwise
        error('Unregonized classmap plot type.');
end

end
