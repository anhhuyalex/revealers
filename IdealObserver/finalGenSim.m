% final simulation
% based on genSortExp1.m
% modified to be simulation rather than human experiment
% Michelle Greene
% November 2013

%clear all;
 rand('state',sum(100*clock));
 %load verySmallWavelets2;
 %testIm = double(imread('livingRoom.jpg'));
 testIm = waveletFeatures(:,3813);
 testIm = testIm'*filters';
 walk_stdev = std(scores)*.05;
 popSize = 100;
 pCross = .4;
 pMut = .3;
 pFreak = .6;
 corGoal = 0.99;
 rejectCor = 0.5646;
 %rejectCor = .1068;
%rejectCor = 0.5729;

converge = 0;
corFlag = 0;
trial = 0;
maxTrials=10000;

while converge == 0
    trial = trial+1
    
    % generate first generation
    if trial == 1
        firstChosen = 0;
        while firstChosen < popSize
            thisPCA = normrnd(mean(scores,1),std(scores,1));
            features = wavecoef(:,1:900)*thisPCA'+meanwavefeatures;
            thisIm = features'*filters';
            [r,p] = corrcoef(thisIm,testIm(:));
            if r(2)<rejectCor
                firstChosen = firstChosen+1;
                pca{trial}(firstChosen,:) = thisPCA;
            end
        end
    end
     if trial==1
         for i = 1:popSize
             thisPCA = normrnd(mean(scores,1),std(scores,1));
             features = wavecoef(:,1:900)*thisPCA'+meanwavefeatures;
             thisIm = features'*filters';
             pca{trial}(i,:) = thisPCA;
         end
     end
    
    % make images
    for i = 1:popSize
        features = wavecoef(:,1:900)*pca{trial}(i,:)'+meanwavefeatures;
        im(:,i) = features'*filters';
    end
    
    % evaluate
    point_totals = zeros(popSize,1);
    
    % sample such that each image competes 5x
    done = 0;
    reject = 0;
    goodTrial = 0;
    rejectThresh = 1000;
    counts = zeros(popSize,1);
    while done==0
        thisPair = randi(popSize,2,1);
        if counts(thisPair(1))>=5 || counts(thisPair(2))>=5
            reject = reject+1;
            if reject==rejectThresh
                break
            end
        else
            goodTrial = goodTrial+1;
            counts(thisPair(1)) = counts(thisPair(1))+1;
            counts(thisPair(2)) = counts(thisPair(2))+1;
            thesePairs(goodTrial,:) = thisPair;
        end     
    end
    
    % evaluate each of the trials
    for i = 1:length(thesePairs)
        im1 = im(:,thesePairs(i,1));
        im2 = im(:,thesePairs(i,2));
        [r,p]=corrcoef(testIm,im1);
        cor1 = r(2);
        [r,p]=corrcoef(testIm,im2);
        cor2 = r(2);
               
        % increment point totals based on response
        if cor1>cor2
            point_totals(thesePairs(i,1)) = point_totals(thesePairs(i,1))+1;
            thisMaxCor = cor1;
        elseif cor2>cor1
            point_totals(thesePairs(i,2)) = point_totals(thesePairs(i,2))+1;
            thisMaxCor = cor2;
        end
    end
    
    % get the next generation: (1) roulette wheel mating
    for i = 1:popSize
        bag = repmat(i,[1 (point_totals(i)-1)^2]);
        if i==1
            bagofparents = bag;
        else bagofparents = cat(2,bagofparents,bag);
        end
    end
    
    order = randperm(length(bagofparents));
    bagofparents = bagofparents(order);
    bagofparents = bagofparents(1:popSize);
    
    for i = 1:length(bagofparents)
        pca{trial+1}(i,:) = pca{trial}(bagofparents(i),:);
    end
    
    % get the next generation: (2) crossover
    for n = 1:popSize
        if rand<pCross
            partner = bagofparents(randi(popSize));
            parent1pcs = pca{trial+1}(n,:);
            parent2pcs = pca{trial+1}(partner,:);
            
            lengthx=0;
            lengthy=0;
            
            for a = 1:900
                if mod(a,2)==0
                    lengthy = lengthy+1;
                    yy(lengthy)=a;
                else
                    lengthx = lengthx+1;
                    xx(lengthx)=a;
                end
            end
            
            pca{trial+1}(n,xx) = parent1pcs(xx);
            pca{trial+1}(n,yy) = parent2pcs(yy);
        end
    end
    
    % get the next generation: (3) mutation
    for n = 1:popSize
        if rand<pMut
            pca{trial+1}(n,:) = normrnd(pca{trial}(n,:),walk_stdev);
        end
    end
    
    % get the next generation: (4) migration
    for n = 1:popSize
        if rand<pFreak
            pca{trial+1}(n,:) = normrnd(mean(scores,1),std(scores,1));
        end
    end
    
    % compute population correlation with test image
    for i = 1:popSize
        thisIm = im(:,i);
        [r,p] = corrcoef(thisIm,testIm(:));
        popCor{trial}(i) = r(2);
    end
    
    % check for convergence
    maxCor = max(popCor{trial});
    if maxCor>corGoal
        corFlag=1;
    end
    
    allPoints(:,trial) = point_totals;
    pFreak = pFreak/2;
    
    % check for convergence
    if corFlag
        converge=1;
    end

if trial==maxTrials
converge=1;
end
    
end
