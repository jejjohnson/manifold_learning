function [svm_stats, lda_stats, fig_handles] = ...
    classification_exp(embedding, gt, options)

switch lower(options.type)
    case 'classification'
        
        
        % find the number of components
        n_components = size(embedding, 2);
        
        % find the number of test_dimensions
        test_dims = (1:10:n_components);
        
        
        % choose training and testing amount
        rng('default');                     % reproducibility
        
        % initialize classifcation storages
        lda_OA = []; lda_AA = []; lda_k = [];
        svm_OA = []; svm_AA = []; svm_k = [];
        
        h = waitbar(0, 'Initializing waitbar...');
        
        for dim = test_dims
            
            
            % number of dimensions
            XS = embedding(:, 1:dim);
            
            % training and testing samples
            [X_train, y_train, X_test, y_test] = train_test_split(...
                XS, gt, options);
            
            % classification SVM
            waitbar(dim/n_components, h, ...
                ['Performing SVM classification, ' options.algorithm]);
            
            [y_pred] = svmClassify(X_train, y_train, X_test);
            
            % obtain classification metrics
            [~, stats] = class_metrics(y_test, y_pred);
            
            % store values for later
            svm_OA = [svm_OA; stats.OA];
            svm_AA = [svm_AA; stats.AA];
            svm_k = [svm_k; stats.k];
            
            % Classification - LDA
            
            waitbar(dim/n_components, h, ...
                ['Performing LDA classification, ' options.algorithm]);
            
            lda_obj = fitcdiscr(X_train, y_train);
            y_pred = predict(lda_obj, X_test);
            
            % obtain classification metrics
            [~, stats] = class_metrics(y_test, y_pred);
            
            % store stats
            lda_OA = [lda_OA; stats.OA];
            lda_AA = [lda_AA; stats.AA];
            lda_k = [lda_k; stats.k];
            
        end
        
        close(h);
        
        % store stats for later
        svm_stats = [svm_OA, svm_AA, svm_k];
        lda_stats = [lda_OA, lda_AA, lda_k];
        
        
        % Classification Plots
        fig_handle = figure('Units', 'pixels', ...
            'Position', [100 100 500 375]);
        hold on;
        
        % plot lines
        hLDA = line(test_dims, lda_OA);
        hSVM = line(test_dims, svm_OA);
        
        % set some first round of line parameters
        set(hLDA, ...
            'Color',        'r', ...
            'LineWidth',    2);
        set(hSVM, ...
            'Color',        'b', ...
            'LineWidth',    2);
        
        % set title and labels
        hTitle = title([options.algo ' + LDA, SVM - ' ...
            options.dataset]);
        hXLabel = xlabel('d-Dimensions');
        hYLabel = ylabel('Correct Rate');
        
    otherwise
        error('Unknown type of experiment.');


end