function [varargout] = plotclassmaps(predImage, originalImage, gtImage, ...
    Options)

if ~isfield(Options, 'imgColors')
    Options.imgColors = [29 15 12];
end


switch lower(Options.type)
    
    case 'learning'
        
        % Display Original Image
        imgColor = originalImage(:,:, Options.imgColors);
        iCMin = min(imgColor(:));
        iCMax = max(imgColor(:));
        imgColor = uint8(255*(imgColor -iCMin)./(iCMax - iCMin));
        imgColorPad = padarray(imgColor, [1 1], 0);
        
        hOriginalImage = figure;
        
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
        
        alphaData = padarray(double(gtMask)*(3/4) + 1/4, [1 1], 1);
        
        
        % Display Resulting Class Labels
        imgClasses = uint8(255*ind2rgb(predImage, hsv(nClasses)));
        imgClassesPad = padarray(imgClasses, [1 1], 0);
        
        hpredImage = figure;
        hpred = imshow(imgClassesPad);
        set(hpred, 'alphaData', alphaData);
    
    case 'alignment'
        % Display Original Image
        imgColor = originalImage{Options.domain}(:,:, Options.ImgColors);
        iCMin = min(imgColor(:));
        iCMax = max(imgColor(:));
        imgColor = uint8(255*(imgColor -iCMin)./(iCMax - iCMin));
        imgColorPad = padarray(imgColor, [1 1], 0);
        
        hOriginalImage = figure;
        
        himg = imshow(imgColorPad);
        
        % Display Ground Truth Image
        gtMask = gtImage{Options.domain} > 0;
        nClasses = numel(unique(gtMask>0));
        gtClasses = uint8(255* ind2rgb(gtImage{Options.domain}, hsv(nClasses))) .* ...
            repmat(uint8(gtMask), [1 1 3]) + ...
            255 * repmat(uint8(~gtMask), [1 1 3]);
        gtClassesPad = padarray(gtClasses, [1 1 0]);
        
        hgtImage = figure;
        hgt = imshow(gtClassesPad);
        
        alphaData = padarray(double(gtMask)*(3/4) + 1/4, [1 1], 1);
        
        
        % Display Resulting Class Labels
        imgClasses = uint8(255*ind2rgb(predImage{Options.domain}, hsv(nClasses)));
        imgClassesPad = padarray(imgClasses, [1 1], 0);
        
        hpredImage = figure;
        hpred = imshow(imgClassesPad);
        set(hpred, 'alphaData', alphaData);
        
    otherwise
        error('Unregonized classmap plot type.');
end

end