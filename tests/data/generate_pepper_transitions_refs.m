% Opt.Transitions (user transition selection) in pepper: S=3/2 and S=1, powder,
% crystal and frequency sweep. Reference for torchspin/tests/test_pepper_transitions_matlab_validation.py.
clear all; cases = {};
function c = mk(name,Sys,Exp,Opt)
  Opt.Verbosity = 0; [x,spc] = pepper(Sys,Exp,Opt);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'x',x,'spc',spc);
end
Sys = struct('S',3/2,'g',2,'D',[1500 200],'lwpp',2);
Exp = struct('mwFreq',9.5,'Range',[0 600],'nPoints',1024);
cases{end+1} = mk('quartet_default',Sys,Exp,struct());
cases{end+1} = mk('quartet_central',Sys,Exp,struct('Transitions',[2 3]));
cases{end+1} = mk('quartet_outer',Sys,Exp,struct('Transitions',[2 1; 4 3]));
cases{end+1} = mk('quartet_all',Sys,Exp,struct('Transitions','all'));
SysS = struct('S',3/2,'g',[2 2.05 2.1],'D',[1500 200],'HStrain',[40 40 40]);
cases{end+1} = mk('quartet_strain_central',SysS,Exp,struct('Transitions',[2 3]));
Sys1 = struct('S',1,'g',2,'D',[30 0],'lwpp',1);
Exp1 = struct('mwFreq',9.5,'Range',[150 190],'nPoints',1024,'Harmonic',0);
cases{end+1} = mk('triplet_halffield_all',Sys1,Exp1,struct('Transitions','all'));
cases{end+1} = mk('triplet_halffield_13',Sys1,Exp1,struct('Transitions',[1 3]));
ExpC = struct('mwFreq',9.5,'Range',[0 600],'nPoints',1024,'CrystalSymmetry','P1','MolFrame',[0.3 0.7 0.1]);
cases{end+1} = mk('quartet_crystal_central',Sys,ExpC,struct('Transitions',[2 3]));
SysF = struct('S',1,'g',2,'D',[3000 300],'lwpp',10);
ExpF = struct('Field',340,'mwRange',[5 15],'nPoints',1024);
cases{end+1} = mk('triplet_freq_12',SysF,ExpF,struct('Transitions',[1 2]));
script_dir = fileparts(mfilename('fullpath')); save(fullfile(script_dir,'ref_pepper_transitions.mat'),'cases','-v7'); fprintf('Done: %d cases\n',numel(cases));
