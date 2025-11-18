function win = kbfiltNd(sz,alpha,dim)
  % 
  % Create a kaiser-bessel filter
  %
  % Inputs
  %   sz:     the size of the filter. e.g., sz = [128 128] for a 2D filter
  %   alpha:  kaiser-bessel alpha parameter. 1 = small; 5 = large
  %   dim:    no filtering along dims of k>dim. 
  %             Filter is just replicated along extra dimensions
  %
  % (c) Corey Baron 2018
  %

  win = kaiser(sz(1),alpha);
  for n=2:dim
    k2 = permute(kaiser(sz(n),alpha), [2:n, 1]);
    win = repmat(win, [ones(1,n-1), sz(n)]) .* repmat(k2, [sz(1:n-1), 1]);
  end
  if dim<length(sz)
    win = repmat(win, [ones(1,dim), sz(dim+1:end)]);
  end

end
