function ece558_hw06_4

N = 32; img = phantom(N); sig2 = 0.1;

Nang = [32,8,32]; Npry = [32,128,32];
Tang = [90,180,180]; Nid = length(Nang);
lab = {'LA','SD','CD'};
alpha = [6.0,1.2,3.0];

for id = 1:Nid
	
	Na = Nang(id); Np = Npry(id); Ta = Tang(id);
	ang = Ta*(0:Na-1)/Na;

	[T,Xp] = projmtx(N,ang,Np);
	
	proj1 = T*reshape(img,N^2,1);
	proj2 = proj1 + sqrt(sig2)*randn([Np*Na,1]);
	
	[U,S,V] = svd(full(T)); s = diag(S);
	tol = max(size(T))*eps(max(s)); Nr = sum(s>tol);
	
	% Picard Analysis
	figure(1);
	if id==1, clf; colormap jet;
		set(gcf,'Position',[0,0,600,800],'PaperPositionMode','auto');
	end;
	
	subplot(3,Nid,id)
	eta = picard(U,s,proj1,0);
	xlim([0,N*N]); ylim([10^-40,10^20]);
	title([lab{id},' case - Rank = ',num2str(Nr)]);
	
	subplot(3,Nid,id+Nid)
	eta = picard(U,s,proj2,0);
	xlim([0,N*N]); ylim([10^-40,10^40]);
	title([lab{id},' case (Data + Noise)']);
	
	subplot(3,Nid,id+2*Nid)
	ind = [1,50,100,150];
	imagesc(abs(V(:,ind)));
	title(['V_{i} vectors (SVD)']);
	
	% SVD Reconstruction
	U1 = U(:,1:Nr); S1 = S(1:Nr,1:Nr); V1 = V(:,1:Nr); K = abs(s(1)/s(Nr));
	new_img1 = reshape(V1*inv(S1)*U1'*proj1,N,N);
	new_img2 = reshape(V1*inv(S1)*U1'*proj2,N,N);
	
	figure(2);
	if id==1, clf; colormap jet;
		set(gcf,'Position',[0,0,600,800],'PaperPositionMode','auto');
	end;
	
	subplot(4,Nid,id);
	imagesc(reshape(proj1,Np,Na)); axis square;
	title(['Projection - ',lab{id},' case']);
	xlabel(['N_{Proj} = ',num2str(Np)]); ylabel(['N_{Ang} = ',num2str(Na)]);
	
	subplot(4,Nid,id+Nid);
	imagesc(new_img1); axis image; title('Pseudo-Inverse');
	
	subplot(4,Nid,id+2*Nid);
	imagesc(reshape(proj2,Np,Na)); axis square;
	title(['Projection + Noise']);
	xlabel(['N_{Proy} = ',num2str(Np)]); ylabel(['N_{Ang} = ',num2str(Na)]);
	
	subplot(4,Nid,id+3*Nid);
	imagesc(new_img2); axis image; title('Pseudo-Inverse + Noise');

	% TSVD Reconstruction
	figure(3);
	if id==1, clf; colormap jet;
		set(gcf,'Position',[0,0,600,900],'PaperPositionMode','auto');
	end;
	subplot(5,Nid,id);
	imagesc(new_img1); axis image; title(['Exact Pseudo-Inverse (',lab{id},' case)']);
	
	subplot(5,Nid,id+Nid);
	imagesc(new_img2); axis image; title(['No regularized (K = ',num2str(K,'%0.2f'),')']);

	subplot(5,Nid,id+2*Nid);
	ind = find(abs(s)>=0.1); K = abs(s(ind(1))/s(ind(end)));
	U1 = U(:,ind); S1 = S(ind,ind); V1 = V(:,ind);
	new_img3 = reshape(V1*inv(S1)*U1'*proj2,N,N);
	imagesc(new_img3); axis image; title(['TSVD ({\alpha} = 0.1, K = ',num2str(K,'%0.2f'),')']);

	subplot(5,Nid,id+3*Nid);
	ind = find(abs(s)>=1E-6); K = abs(s(ind(1))/s(ind(end)));
	U1 = U(:,ind); S1 = S(ind,ind); V1 = V(:,ind);
	new_img3 = reshape(V1*inv(S1)*U1'*proj2,N,N);
	imagesc(new_img3); axis image; title(['TSVD ({\alpha} = 10^{-6}, K = ',num2str(K,'%0.2f'),')']);
	
	subplot(5,Nid,id+4*Nid);
	ind = find(abs(s)>=alpha(id)); K = abs(s(ind(1))/s(ind(end)));
	U1 = U(:,ind); S1 = S(ind,ind); V1 = V(:,ind);
	new_img4 = reshape(V1*inv(S1)*U1'*proj2,N,N);
	imagesc(new_img4); axis image; title(['TSVD ({\alpha} = ',num2str(alpha(id)),', K = ',num2str(K,'%0.2f'),')']);
	
end

figure(1)
print(gcf,'-depsc','ece558_hw06_4a');
set(gcf,'PaperPositionMode','manual');

figure(2)
print(gcf,'-depsc','ece558_hw06_4b');
set(gcf,'PaperPositionMode','manual');

figure(3)
print(gcf,'-depsc','ece558_hw06_4c');
set(gcf,'PaperPositionMode','manual');


function [T,Xp] = projmtx(N,ang,Np)

Na = length(ang);
T = sparse(Na*Np,N^2);
delta = sparse(N,N);

for id = 1:(N^2)
	delta(id) = 1;
	[R,Xp] = radon(full(delta),ang,Np);
	T(:,id) = R(:); delta(id) = 0;
end
