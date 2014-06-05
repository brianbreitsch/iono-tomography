function ece558_hw05_2

N = 32; img = phantom(N);

Nv = [16,32,64,128];

figure(1); clf; colormap jet;
set(gcf,'Position',[0,0,600,800],'PaperPositionMode','auto');

for id = 1:length(Nv)
	Na = Nv(id); Np = Nv(id);
	ang = 180*(0:Na-1)/Na;

	[T,Xp] = projmtx(N,ang,Np);
	proj = T*reshape(img,N^2,1);
	new_img1 = reshape(T.'*proj,N,N);
	new_img2 = reshape(fbp(proj,T,Na),N,N);

	subplot(length(Nv),3,3*id-2);
	imagesc(reshape(proj,Np,Na)); axis square;
	title(['Projection N = ',num2str(Nv(id))]);
	subplot(length(Nv),3,3*id-1);
	imagesc(new_img1); axis image; title('Backprojection');
	subplot(length(Nv),3,3*id);
	imagesc(new_img2); axis image; title('FBP Reconstruction');
end

print(gcf,'-depsc','ece558_hw05_2');
set(gcf,'PaperPositionMode','manual');


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
