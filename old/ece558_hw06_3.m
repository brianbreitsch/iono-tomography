function ece558_hw06_3

N = 32; img = phantom(N);

Nang = [32,8,32]; Npry = [32,128,32];
Tang = [90,180,180]; Nid = length(Nang);
lab = {'LA','SD','CD'};

figure(1); clf; colormap jet;
set(gcf,'Position',[0,0,600,800],'PaperPositionMode','auto');

for id = 1:Nid
	Na = Nang(id); Np = Npry(id); Ta = Tang(id);
	ang = Ta*(0:Na-1)/Na;

	[T,Xp] = projmtx(N,ang,Np);
	
	proj = T*reshape(img,N^2,1);
	new_img1 = reshape(fbp(proj,T,Na),N,N);
	
	[U,S,V] = svd(full(T)); s = diag(S);
	tol = max(size(T))*eps(max(s)); Nr = sum(s>tol);
	U1 = U(:,1:Nr); S1 = S(1:Nr,1:Nr); V1 = V(:,1:Nr);
	new_img2 = reshape(V1*inv(S1)*U1'*proj,N,N);
	
	subplot(4,Nid,id);
	imagesc(reshape(proj,Np,Na)); axis square;
	title(['Projection - ',lab{id},' case']);
	xlabel(['N_{Proj} = ',num2str(Np)]);
	ylabel(['N_{Ang} = ',num2str(Na)]);
	subplot(4,Nid,id+3);
	imagesc(new_img1); axis image; title('FBP Reconstruction');
	subplot(4,Nid,id+6);
	semilogy(diag(S),'.'); axis([0,Nr,10^-4,10^2]);
	axis square; title(['Sing. Values - Rank = ',num2str(Nr)]);
	subplot(4,Nid,id+9);
	imagesc(new_img2); axis image; title('Pseudo-Inverse');
end

print(gcf,'-depsc','ece558_hw06_3');
set(gcf,'PaperPositionMode','manual')


% Filter Back Projection
function P = fbp(Y,T,Na)

Np = length(Y)/Na;

R = pi*(-Np/2:Np/2-1)/(Np/2);
R = fftshift(abs(R))';
R = R*ones(1,Na);

Y = reshape(Y,Np,Na);
P = real(ifft(R.*fft(Y)));
P = T.'*P(:);


function [T,Xp] = projmtx(N,ang,Np)

Na = length(ang);
T = sparse(Na*Np,N^2);
delta = sparse(N,N);

for id = 1:(N^2)
	delta(id) = 1;
	[R,Xp] = radon(full(delta),ang,Np);
	T(:,id) = R(:); delta(id) = 0;
end
