function [] = showOrderedMap_modified(map, path,range)
    sim = map;

    if(nargin < 3)
        Bar_range = [min(map, [], 'all') max(map, [], 'all')];
        Bar_range = round(Bar_range,2);
    else
        Bar_range = range;
    end

    if(nargin < 2)
        path = nan;
    end
    
    FC_input = sim;
    
    % ---------  reordering -----------------------
    tbl = readtable('/data/users4/aballem1/Toolbox/Neuromark2.3FinalPP/Neuromark_fMRI_2-2_labels_final.xlsx');
    T = table2cell(tbl);
    
    ICN_idx = find(strcmp(tbl.Properties.VariableNames,'subdomain_abbrev'));
    temp_idx = find(strcmp(T(:,ICN_idx),'CB'));
    ICN_CB = cell2mat(T(temp_idx,1));
    
    temp_idx = find(strcmp(T(:,ICN_idx),'OT'));
    ICN_VI1 = cell2mat(T(temp_idx,1));
    temp_idx = find(strcmp(T(:,ICN_idx),'OC'));
    ICN_VI2 = cell2mat(T(temp_idx,1));
    
    temp_idx = find(strcmp(T(:,ICN_idx),'PL'));
    ICN_PL = cell2mat(T(temp_idx,1));
    
    temp_idx = find(strcmp(T(:,ICN_idx),'EH'));
    ICN_SC1 = cell2mat(T(temp_idx,1));
    temp_idx = find(strcmp(T(:,ICN_idx),'ET'));
    ICN_SC2 = cell2mat(T(temp_idx,1));
    temp_idx = find(strcmp(T(:,ICN_idx),'BG'));
    ICN_SC3 = cell2mat(T(temp_idx,1));
    
    temp_idx = find(strcmp(T(:,ICN_idx),'SM'));
    ICN_SM = cell2mat(T(temp_idx,1));
    

    temp_idx = find(strcmp(T(:,ICN_idx),'IT'));
    ICN_HC1 = cell2mat(T(temp_idx,1));
    temp_idx = find(strcmp(T(:,ICN_idx),'TP'));
    ICN_HC2 = cell2mat(T(temp_idx,1));
    temp_idx = find(strcmp(T(:,ICN_idx),'FR'));
    ICN_HC3 = cell2mat(T(temp_idx,1));
    
    temp_idx = find(strcmp(T(:,ICN_idx),'CE'));
    ICN_TN1 = cell2mat(T(temp_idx,1));
    temp_idx = find(strcmp(T(:,ICN_idx),'DM'));
    ICN_TN2 = cell2mat(T(temp_idx,1));
    temp_idx = find(strcmp(T(:,ICN_idx),'SA'));
    ICN_TN3 = cell2mat(T(temp_idx,1));
    
    T_ord = T;
    [~,ord_idx] = sort(cell2mat(T_ord(:,1)));
    abbrev = string(T_ord(:,12));
    abbrev = abbrev(ord_idx);
    
    domain_ICN  = {ICN_CB, ICN_VI1, ICN_VI2, ICN_PL, ICN_SC1, ICN_SC2, ICN_SC3, ICN_SM, ICN_HC1, ICN_HC2, ICN_HC3, ICN_TN1, ICN_TN2, ICN_TN3};
    domain_Name = {'CB', 'VI-OT', 'VI-OC', 'PL', 'SC-EH', 'SC-ET', 'SC-BG', 'SM', 'HC-IT', 'HC-TP', 'HC-FR', 'TN-CE', 'TN-DM', 'TN-SA'};
    
    %Draw_FNC_Trends(FC_input, domain, domain_ICN, Bar_range)

    %Draw_FNC(FC_input, domain, domain_ICN, Bar_range)
    show_half_and_half_opt = false;
    showDiagonalEntriesFlag = true;
    drawLinesWithinGridFlag = true;
    show_abbrev = false;
    Draw_FNC_Trends_Mah_V3(FC_input, domain_Name, domain_ICN, Bar_range, show_half_and_half_opt,showDiagonalEntriesFlag, drawLinesWithinGridFlag,abbrev,show_abbrev);
    %Draw_FNC_Trends_Mah_Jet(FC_input, domain_Name, domain_ICN, Bar_range)
    
%     if(~isnan(path))
%         set(gcf,'Color','w');
%         print(gcf, path, '-dpng', '-r600'); % saves as PNG with 600 dpi
% 
%     end


    if(~isnan(path))
        set(gcf,'Color','w');
        print(gcf, path, '-dpng', '-r100'); 

    end
    
    
end
