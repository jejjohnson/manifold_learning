function [C,F,W]=post_cpoint(A,S,C,F,N)

[row,col]=size(A);
old_F=F;
W=sparse(row,col);
T=zeros(1,col);
temp=find(F~=0);
fnum=size(temp,2);
clear temp
temp=find(C~=0);
cnum=size(temp,2);
l=0;
i=0;
while (any(F))
    CT=sparse(1,col);
    CI=sparse(1,col);
    d=sparse(1,col);
    mask=[];
    in_T=1;
    i=i+1;
    while(in_T==1&i<=col)
        mask=find(T==F(i));
        if(~isempty(mask)|F(i)==0)
            i=i+1;
        else
            in_T=0;
            T(i)=F(i);
            break
        end
    end
    if(i>=col) 
        break
    end
    sindx=find(S(F(i),:));
    nindx=find(N(F(i),:));
    [temp,snum]=size(sindx);
    [temp,nnum]=size(nindx);
    if(snum==0)
        disp('There is no strong connection for this fine point')
        cnum=cnum+1;
        C(cnum)=F(i);
        F(i)=0;
    end
    
    m=0;
    dwm=0;
    dsm=0;
    %%% setup Ci,Di_strong==DS %%%
    DS=sparse(1,snum);
    DW=sparse(1,nnum);
    for j=1:snum,
        mask=[];
        mask=find(C==S(F(i),sindx(j)));
        if(~isempty(mask))
            m=m+1;   %%% m = number of elements in CI %%%%
            CI(m)=S(F(i),sindx(j));
            d(CI(m))=A(F(i),CI(m));
        end
    end
    for j=1:snum,
        mask=[];
        mask=find(CI==S(F(i),sindx(j)));
        if(isempty(mask))
            dsm=dsm+1;
            DS(dsm)=S(F(i),sindx(j));
        end
    end
    %%% setup Di_weak==DW
    if(F(i)~=0)
        d(F(i))=A(F(i),F(i));
        for j=1:nnum,
            mask=[];
            mask=find(S(F(i),:)==N(F(i),nindx(j)));
            if(isempty(mask))
                dwm=dwm+1; %%% dwm = number of elements in DW %%%%
                DW(dwm)=N(F(i),nindx(j));
                d(F(i))=d(F(i))+A(F(i),DW(dwm));
            end
        end
    end   
    %%%% start step 5 %%%%
    cm=0;
    done=0;
    while(~done&snum~=0)
        nextf=0;
        for j=1:snum,
            if(DS(j)~=0)
                mask=[];
                for k=1:m,
                    mask=find(S(DS(j),:)==CI(k)); 
                    if(~isempty(mask))            %%% if S_j ^ C_i not empty %%%
                        nextf=nextf+1;
                        sum_ajl=0;
                        for k=1:m,
                            sum_ajl=sum_ajl+A(DS(j),CI(k));
                        end
                        for k=1:m,
                            if(sum_ajl==0)
                                disp('error in post_cpoint: divided by 0 when computing interpolation weights')
                            end
                            d(CI(k))=d(CI(k))+A(F(i),DS(j))*A(DS(j),CI(k))/sum_ajl;
                        end
                        break;
                    end
                end
                if(isempty(mask))               %%% if S_j ^ C_i is empty %%%
                    if(any(CT)==0)
                        cm=cm+1;
                        m=m+1;
                        dsm=dsm-1;
                        CT(cm)=DS(j);
                        CI(m)=DS(j);
                        DS(j)=0;
                        d(CI(m))=A(F(i),CI(m));
                        done=0;
                        break;
                    else
                        cnum=cnum+1;
                        C(cnum)=F(i);
                        F(i)=0;
                        done=1;
                        break;
                    end
                end %if(isempty(mask))  
            elseif(j==snum)
                done=1;
                break;
            else
                done=0;
            end  %if(DS(j)~=0)
            
        end %for j=1:snum
    end %while(~done)
    
    if(F(i)~=0)
        if(any(CT))
            cnum=cnum+1;
            C(cnum)=CT(cm);
            remove_f=find(F==CT(cm));  
            F(remove_f)=0;
        end
        for k=1:m,		
            if(d(F(i))==0)
                disp('error in final c-point selection: interpolation weight are not well defined')
            end
            W(F(i),CI(k))=-d(CI(k))/d(F(i));
        end
    end
    if(T==old_F)
        break
    end
end %%% end while loop %%%



