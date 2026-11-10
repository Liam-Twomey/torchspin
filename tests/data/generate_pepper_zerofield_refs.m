% Zero-field frequency sweeps (torchspin Options.ZeroField, 0.7.4): EasySpin
% pepper accepts Exp.Field = 0 for frequency sweeps (validate_exp: Field < 0
% is rejected, 0 is not) and resolves zero-field degeneracies in gethamdata
% (B == 0: eigenvectors of muzL inside each degenerate block).
% Saved to ref_pepper_zerofield.mat (-v7) as a cell array `cases`; each case
% stores name, Sys, Exp, Opt and the spectrum (nu, spc).
clear all
fprintf('Generating pepper zero-field references...\n');
cases = {};

% S = 1, rhombic D: lines at D-E, D+E, 2E
Sys = struct('S',1,'g',2,'D',[3000 400],'lw',[5 0]);
Exp = struct('Field',0,'mwRange',[0.1 4],'nPoints',4000,'Harmonic',0);
cases{end+1} = mk('s1_DE', Sys, Exp, struct());
% same, Boltzmann populations at 0.1 K
Exp_T = Exp; Exp_T.Temperature = 0.1;
cases{end+1} = mk('s1_DE_temp', Sys, Exp_T, struct());
% same, direction-dependent HStrain (MHz) instead of lw
Sys_H = rmfield(Sys,'lw'); Sys_H.HStrain = [10 20 30];
cases{end+1} = mk('s1_DE_hstrain', Sys_H, Exp, struct());
% S = 3/2 Kramers system: one line at 2*sqrt(D^2+3E^2) between the two doublets
Sys = struct('S',3/2,'g',2,'D',[5000 1000],'lw',[20 0]);
Exp = struct('Field',0,'mwRange',[9 12],'nPoints',3000,'Harmonic',0);
cases{end+1} = mk('s32_kramers', Sys, Exp, struct());
% S = 1/2, I = 1/2 isotropic hyperfine (hydrogen-like F=0 <-> F=1 line at A)
Sys = struct('S',1/2,'g',2,'Nucs','1H','A',1420,'lw',[2 0]);
Exp = struct('Field',0,'mwRange',[1.3 1.5],'nPoints',2001,'Harmonic',0);
cases{end+1} = mk('hyd_iso', Sys, Exp, struct());

script_dir = fileparts(mfilename('fullpath'));
save(fullfile(script_dir,'ref_pepper_zerofield.mat'),'cases','-v7');
fprintf('Done: %d cases\n',numel(cases));

function c = mk(name,Sys,Exp,Opt)
  Opt.Verbosity = 0;
  [nu,spc] = pepper(Sys,Exp,Opt);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'nu',nu,'spc',spc);
end
