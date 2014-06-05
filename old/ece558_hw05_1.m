function ece558_hw05_1

N = 32; Na = 64; Np = 64;
img = phantom(N);
ang = 180*(0:Na-1)/Na;

[T,Xp] = projmtx(N,ang,Np);
proj = T*reshape(img,N^2,1);
new_img = reshape(T.'*proj,N,N);

figure(1); clf; colormap jet;

subplot(2,2,1);
imagesc(img); axis image; title(['Phantom Image (',num2str(N),'x',num2str(N),')']);

subplot(2,2,2);
spy(T); axis square; title('T Matrix');

subplot(2,2,3);
imagesc(ang,Xp,reshape(proj,Np,Na)); axis square;
xlabel('Angle \theta (deg)');
title(['Projection (N_{\theta} = ',num2str(Na),', N_{P} = ',num2str(Np),')']);

subplot(2,2,4);
imagesc(new_img); axis image; title(['Backprojection (',num2str(N),'x',num2str(N),')']);
print(gcf,'-depsc','ece558_hw05_1a');

figure(2); clf;

N = [4,8,16];
for id = 1:length(N)
	ang = 180*(0:N(id)-1)/N(id);
	T = projmtx(N(id),ang,N(id));
	subplot(2,2,id);
	null_mat = null(full(T));
	null_sz = length(null_mat(1,:));
	spy(T); title(['T Matrix (N = ',num2str(N(id)),', Nulls = ',num2str(null_sz),')']);
end

subplot(2,2,4); N = N(end);
for id = 1:null_sz
	img = reshape(null_mat(:,id),N,N);
	imagesc(img); colormap gray; axis image;
	title(['Null space N = ',num2str(N)]); drawnow;
end

print(gcf,'-depsc','ece558_hw05_1b');


% Projection Matrix
function [T,Xp] = projmtx(N,ang,Np)

Na = length(ang);
T = sparse(Na*Np,N^2);
delta = sparse(N,N);

for id = 1:(N^2)
	delta(id) = 1;
	[R,Xp] = radon(full(delta),ang,Np);
	T(:,id) = R(:); delta(id) = 0;
end
