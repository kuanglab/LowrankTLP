function [] = write_output(S, eigs, Q, G, f)

% inputs
dlmwrite('S.tns', S, 'delimiter', ' ');

dlmwrite('eigs.txt', eigs, 'delimiter', ' ', 'precision', 16);
for m = 1:size(Q, 1);
  fname = sprintf('q%d.txt', m);
  dlmwrite(fname, Q{m}, 'delimiter', ' ', 'precision', 16);
end

% outputs
dlmwrite('gold_predicted.txt', G, 'delimiter', ' ', 'precision', 16);
dlmwrite('gold_f.txt', f, 'delimiter', ' ', 'precision', 16);

