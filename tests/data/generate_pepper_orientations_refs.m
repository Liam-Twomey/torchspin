% EasySpin references for torchspin's Exp.Orientations (user-supplied field
% directions in the molecular frame, chi-averaged rates; torchspin 0.7.6).
% EasySpin has no orientation list for disordered samples (its crystal
% orientations use B1 along lab x), so each case emulates the chi average with
% a crystal simulation: Exp.MolFrame = [0 0 0] (sample = molecular frame),
% one Exp.SampleFrame row [0 theta pi-phi] per orientation (puts lab z = B0
% along (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)) in the molecular
% frame) and Exp.SampleRotation = {'z',[0 pi/2]}: two B1 directions
% perpendicular to B0, whose mean rate is the exact chi average.  EasySpin
% weights every crystal orientation by 1/(nSamples*nRotations), which equals
% torchspin's default weights 1/N.
clear all
cases = {};
function c = mk(name,Sys,Exp,Opt,ori)
  Opt.Verbosity = 0;
  Exp.MolFrame = [0 0 0];
  Exp.SampleFrame = [zeros(size(ori,1),1) ori(:,2) pi-ori(:,1)];
  Exp.SampleRotation = {'z',[0 pi/2]};
  [x,spc] = pepper(Sys,Exp,Opt);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'ori',ori,'x',x,'spc',spc);
end
% 1. rhombic g, tilted 14N hyperfine, Voigt lw, three orientations, 1st harmonic
Sys = struct('g',[2.0 2.05 2.2],'gFrame',[0.2 0.4 0.1],'Nucs','14N','A',[20 30 90],'lw',[0.7 0.4]);
Exp = struct('mwFreq',9.5,'Range',[290 350],'nPoints',600,'Harmonic',1);
cases{end+1} = mk('nitro_voigt',Sys,Exp,struct(),[0.3 0.7; 1.9 2.2; 4.0 1.1]);
% 2. HStrain, single orientation, absorption
Sys = struct('g',[2.0 2.1 2.2],'HStrain',[30 60 90]);
Exp = struct('mwFreq',9.5,'Range',[290 350],'nPoints',800,'Harmonic',0);
cases{end+1} = mk('hstrain_single',Sys,Exp,struct(),[0.8 1.0]);
% 3. triplet with D and Boltzmann populations, four orientations, Lorentzian lw
Sys = struct('S',1,'g',2.0,'D',[900 150],'lw',[0 1.5]);
Exp = struct('mwFreq',9.5,'Range',[250 430],'nPoints',900,'Harmonic',1,'Temperature',20);
cases{end+1} = mk('triplet_T',Sys,Exp,struct(),[0 0.2; 0.5 0.9; 1.2 1.5; 2.5 2.4]);
% 4. frequency sweep (GHz axis, lw in MHz), two orientations
Sys = struct('g',[2.0 2.05 2.2],'lw',[20 0]);
Exp = struct('Field',330,'mwRange',[9.0 9.9],'nPoints',700,'Harmonic',0);
cases{end+1} = mk('freqsweep',Sys,Exp,struct(),[0.4 0.6; 2.0 1.3]);
% 5. axial g: Fibonacci quadrature of the sphere (crystal sums) vs the powder
n = 4000; i = (0:n-1)' + 0.5; th = acos(1-2*i/n); ph = mod(pi*(1+sqrt(5))*i,2*pi);
Sys = struct('g',[2.0 2.0 2.2],'lw',[1 0]);
Exp = struct('mwFreq',9.5,'Range',[300 350],'nPoints',1024,'Harmonic',0);
cases{end+1} = mk('axial_fibonacci',Sys,Exp,struct(),[ph th]);
[x,spc] = pepper(Sys,Exp,struct('Verbosity',0));
cases{end+1} = struct('name','axial_powder','Sys',Sys,'Exp',Exp,'Opt',struct(),'ori',[],'x',x,'spc',spc);
dx = (Exp.Range(2)-Exp.Range(1))/(Exp.nPoints-1); Exp.Range = Exp.Range - dx/2;
[x,spc] = pepper(Sys,Exp,struct('Verbosity',0));
cases{end+1} = struct('name','axial_powder_halfbin','Sys',Sys,'Exp',Exp,'Opt',struct(),'ori',[],'x',x,'spc',spc);
script_dir = fileparts(mfilename('fullpath'));
save(fullfile(script_dir,'ref_pepper_orientations.mat'),'cases','-v7');
fprintf('Done: %d cases\n',numel(cases));
