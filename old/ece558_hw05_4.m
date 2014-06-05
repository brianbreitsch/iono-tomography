function ece558_hw05_4

N = 32; img = phantom(N);

Nv = [16,32,64,128];
mode = [0,1];

for im = 1:length(mode)

	figure(im); clf; colormap jet;
	set(gcf,'Position',[0,0,600,800],'PaperPositionMode','auto');

	for id = 1:length(Nv)
		Na = Nv(id); Np = Nv(id);
		ang = 180*(0:Na-1)/Na;

		[T,Xp] = projmtx(N,ang,Np);
		proj = T*reshape(img,N^2,1);
		new_img1 = reshape(T.'*proj,N,N);
		X0 = new_img1;
		new_img2 = art(proj,T,X0,mode(im));

		subplot(length(Nv),3,3*id-2);
		imagesc(reshape(proj,Np,Na)); axis square;
		title(['Projection N = ',num2str(Nv(id))]);
		subplot(length(Nv),3,3*id-1);
		imagesc(new_img1); axis image; title('Backprojection');
		subplot(length(Nv),3,3*id);
		imagesc(new_img2); axis image; title('FBP Reconstruction');
	end

	print(gcf,'-depsc',['ece558_hw05_4',num2str(im)]);
	set(gcf,'PaperPositionMode','manual');

end


% Algebraic Reconstruction Technique
function X = art(Y,T,X0,mode)

if ~exist('mode','var'), mode = 0; end;
[M,N] = size(X0); X = reshape(X0,M*N,1);
T = full(T); ST2 = sum(T.^2,2); ind = find(ST2~=0);

for m = 1:2
	for n = 1:length(ind)
		X = X + (Y(ind(n)) - T(ind(n),:)*X)/ST2(ind(n))*T(ind(n),:).';
		if mode, X = max(X,0); end;
	end
end
X = reshape(X,M,N);


function [T,Xp] = projmtx(N,ang,Np)

Na = length(ang);
T = sparse(Na*Np,N^2);
delta = sparse(N,N);

for id = 1:(N^2)
	delta(id) = 1;
	[R,Xp] = radon(full(delta),ang,Np);
	T(:,id) = R(:); delta(id) = 0;
end
