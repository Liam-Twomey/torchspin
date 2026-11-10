% Generate EasySpin references for ENDOR hyperfine enhancement
% (Opt.Enhancement = 'on'/'off') in endorfrq and salt.
% Run in MATLAB with EasySpin on the path from the tests/data/ directory.
%
%   >> cd tests/data
%   >> generate_endor_enhancement_refs
%
% Saves endor_enhancement_refs.mat with struct `ref`:
%   ref.endorfrq{k}  single-orientation endorfrq, powder (chi-averaged) mode:
%       .Sys, .Field, .phi, .theta,
%       .Tr      transitions (1-based level pairs, ascending energy)
%       .Pon/.Ion, .Poff/.Ioff  positions (MHz) and intensities
%   ref.salt{k}      fixed-field salt spectra:
%       .Sys, .Exp, .Opt, .x, .yon, .yoff
%
% endorfrq is called in its powder mode (as salt calls it): orientation
% passed in Exp.MolFrame = [0 -theta -phi] with Opt.crystalSample = false,
% so the rf operator is chi-averaged over lab x and y, as in torchspin.

clear ref
ref.endorfrq = {};
ref.salt = {};

% -------------------------------------------------------------------------
% endorfrq cases
% -------------------------------------------------------------------------
cases = {};

c = struct();  % S=1/2 + 1H, rhombic A (EasySpin tests/endorfrq_enhancement.m)
c.Sys = struct('S',1/2,'g',[2 2 2],'Nucs','1H','A',[3 7 12]);
c.Field = 350;
cases{end+1} = c;

c = struct();  % nitroxide-like: 14N (with Q) + 1H
c.Sys = struct('S',1/2,'g',[2.0088 2.0061 2.0027],'Nucs','14N,1H', ...
  'A',[17 17 98; 2 3 5],'Q',[1.0 0.5 -1.5; 0 0 0]);
c.Field = 340;
cases{end+1} = c;

c = struct();  % Cu(II)-like: 63Cu + 14N, large hyperfine
c.Sys = struct('S',1/2,'g',[2.05 2.06 2.25],'Nucs','63Cu,14N', ...
  'A',[60 70 550; 40 42 50],'Q',[-3 -3 6; 0.4 0.6 -1.0]);
c.Field = 300;
cases{end+1} = c;

angles = [0 0; 0.3 0.7; 1.1 1.3; pi/5 pi/7];   % [phi theta] rows

for ic = 1:numel(cases)
  for ia = 1:size(angles,1)
    phi = angles(ia,1); theta = angles(ia,2);
    Exp = struct('Field',cases{ic}.Field,'MolFrame',[0 -theta -phi]);
    Opt = struct('crystalSample',false,'rotatedSample',false, ...
                 'Threshold',1e-4);
    Opt.Enhancement = 'off';
    [Poff,Ioff,Tr] = endorfrq(cases{ic}.Sys,Exp,Opt);
    Opt.Enhancement = 'on';
    [Pon,Ion,Tr2] = endorfrq(cases{ic}.Sys,Exp,Opt);
    assert(isequal(Tr,Tr2));
    r = struct('Sys',cases{ic}.Sys,'Field',cases{ic}.Field, ...
      'phi',phi,'theta',theta,'Tr',Tr,'Pon',Pon,'Ion',Ion, ...
      'Poff',Poff,'Ioff',Ioff);
    ref.endorfrq{end+1} = r;
  end
end

% -------------------------------------------------------------------------
% salt cases (fixed field)
% -------------------------------------------------------------------------
% 1) EasySpin tests/salt_enhancement.m system (14N with Q, HStrain)
Sys = struct('Nucs','14N','g',[2.25 2.25 2],'A',[44,49,54], ...
  'Q',[1 1 -2]*0.84,'lwEndor',0.5);
Exp = struct('Range',[16 65],'Field',326.5,'nPoints',2001,'mwFreq',9.7);
Opt = struct('Threshold',1e-4,'GridSize',[31 6]);
ref.salt{end+1} = run_salt(Sys,Exp,Opt);

% 2) S=1/2 + 1H with large rhombic A (enhancement visible between branches)
Sys = struct('S',1/2,'g',[2 2.01 2.02],'Nucs','1H','A',[6 9 16], ...
  'lwEndor',0.2);
Exp = struct('Range',[2 25],'Field',326.5,'nPoints',2001,'mwFreq',9.7);
Opt = struct('Threshold',1e-4,'GridSize',[31 6]);
ref.salt{end+1} = run_salt(Sys,Exp,Opt);

% 3) nitroxide-like 14N (with Q) + 1H
Sys = struct('S',1/2,'g',[2.0088 2.0061 2.0027],'Nucs','14N,1H', ...
  'A',[17 17 98; 2 3 5],'Q',[1.0 0.5 -1.5; 0 0 0],'lwEndor',0.3);
Exp = struct('Range',[0 70],'Field',340,'nPoints',2001,'mwFreq',9.55);
Opt = struct('Threshold',1e-4,'GridSize',[31 6]);
ref.salt{end+1} = run_salt(Sys,Exp,Opt);

save('endor_enhancement_refs.mat','ref','-v7');
fprintf('saved endor_enhancement_refs.mat (%d endorfrq, %d salt cases)\n', ...
  numel(ref.endorfrq), numel(ref.salt));

function r = run_salt(Sys,Exp,Opt)
  Opt.Enhancement = 'off';
  [x,yoff] = salt(Sys,Exp,Opt);
  Opt.Enhancement = 'on';
  [~,yon] = salt(Sys,Exp,Opt);
  r = struct('Sys',Sys,'Exp',Exp,'Opt',Opt,'x',x(:)','yon',yon(:)','yoff',yoff(:)');
end
