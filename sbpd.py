import numpy as np
import scipy.signal as sg

########################################################################
#       General functions used in the algorithm
########################################################################


#argdict holds the "global" arguments/parameters
#
#
# 'musol' - dual solution
# 'xsol' - primal solution
# 'loopcounter' - object to iterate over, either range(itera) or progress bar version
# 'lagrangian' - boolean to say whether or not to compute lagrangian values
# 'primalsize' - step size for primal
# 'dualsize' - step size for dual
# 'infballrad' - radius of inf-norm ball for TV problems
# 'A' - matrix A in KL divergence for linear inverse problems on simplex
# 'y' - vector y in KL divergence for linear inverse problems on simplex
# 'linopTmusol' - precomputed adjoint of the linear operator applied to dual solution
# 'linopxsol' - precomputed linear operator applied to the primal solution
# 'fxsol' - precomputed smooth primal function applied to the primal solution
# 'Alist' - list of linear operators in trend filtering
# 'theta' - reference measure in Wasserstein inverse problem
# 'dimension' - dimension of the reference measure in Wasserstein inverse problem
# 'gamma' - regularization strength for entropically regularized Wasserstein
# 'costmatrix' - ground cost matrix for optimal transport
# 'hmusol' - precomputed smooth dual function applied to the dual solution
# 'bumpF' - Real Fourier transform (np.fft.rfft) of the bump function in Wasserstein inverse problem
# 'bumpFconj' - conjugate of bumpF
# 'ghat' - the fourier transform of the function generating the ground cost matrix
# 'padlength' - the number of zeros to pad for convolution
#
#
# BARYCENTER ---
# 'dimension1'
# 'dimension2'
# 'theta1'
# 'theta2'
# 'epsilon1'
# 'epsilon2'
# 'costmatrix1'
# 'costmatrix2'
# 'alpha1'
# 'alpha2'
# 'bump1F'
# 'bump2F'
# 'bump1Fconj'
# 'bump2Fconj'

def ibpd(itera, x_initial, mu_initial, primal_map, dual_map, primal_prox, dual_prox, nabf, nabh, linop, linopT, lagr_x, lagr_mu, argdict):
	x = x_initial
	mu = mu_initial
	ergx = np.zeros(x_initial.shape)
	ergmu = np.zeros(mu_initial.shape)
	erglagr_vals = np.zeros(itera)
	pwlagr_vals = np.zeros(itera)
	musol = argdict['musol']
	xsol = argdict['xsol']
	loopcounter = argdict['loopcounter']
	if argdict['stochastic']:
		for i in loopcounter:
			xpast = x
			x = stoch_primal_step(x, mu, primal_map, primal_prox, nabf, linopT, argdict)
			mu = dual_step(x, xpast, mu, dual_map, dual_prox, nabh, linop, argdict)
			if argdict['lagrangian'] == 1:
				ergx = 1. / (i + 1) * x + float(i) / (i + 1) * ergx
				ergmu = 1. / (i + 1) * mu + float(i) / (i + 1) * ergmu
				erglagr_vals[i] = lagr_x(ergx, argdict) - lagr_mu(ergmu, argdict)
				pwlagr_vals[i] = lagr_x(x, argdict) - lagr_mu(mu, argdict)
	else:
		for i in loopcounter:
			xpast = x
			x = primal_step(x, mu, primal_map, primal_prox, nabf, linopT, argdict)
			mu = dual_step(x, xpast, mu, dual_map, dual_prox, nabh, linop, argdict)
			if argdict['lagrangian'] == 1:
				ergx = 1. / (i + 1) * x + float(i) / (i + 1) * ergx
				ergmu = 1. / (i + 1) * mu + float(i) / (i + 1) * ergmu
				erglagr_vals[i] = lagr_x(ergx, argdict) - lagr_mu(ergmu, argdict)
				pwlagr_vals[i] = lagr_x(x, argdict) - lagr_mu(mu, argdict)
	return x, mu, erglagr_vals, pwlagr_vals
	
def primal_step(x, mu, primal_map, primal_prox, nabf, linopT, argdict):
	lam = argdict['primalsize']
	u = primal_map(x, argdict) - lam * (nabf(x, argdict) + linopT(mu, argdict))
	return primal_prox(u, argdict)
	
def dual_step(x, xpast, mu, dual_map, dual_prox, nabh, linop, argdict):
	nu = argdict['dualsize']
	w = dual_map(mu) - nu * (nabh(mu, argdict) - linop((2 * x) - xpast, argdict))
	return dual_prox(w, argdict)

########################################################################
#     These functions are used for multiple different problems
########################################################################


#identity map
#use this when you are in euclidean space
def id_map(mu, argdict=None):
	return mu

#zero map
#use this when there is no smooth term, e.g. when gradf or gradh are identically 0
def zero_map(mu, argdict=None):
	return np.zeros(mu.shape)

#the KL divergence
#used to calculate the constant in the ergodic rate
def kldiv(x,y):
	np.seterr(divide = 'ignore') 
	logxovery = np.where(x / y != 0, np.log(x / y), x / y)
	np.seterr(divide = 'warn') 
	return (x * logxovery - x + y).sum()

#gradient of the shannon-boltzmann entropy
def shanbolt_grad(x, argdict=None):
	return np.log(x) + 1

#project onto the affine constraint under the KL geometry.
#we use the "shift trick" by subtracting the max, in hopes of avoiding overflow error
#
#if x is 1-dimensional it is the normal exponential regularize i.e.
#it is the bregman projection onto the affine constraint under 
#shannon-boltzman entropy
#
#if x is 2-dimensional, it is the 1-dimensional exponential regularize
#applied to each row (not something joint in the rows/columns)
def exp_regularize(x, argdict=None):
	if len(x.shape) == 2:
		rowmax = np.expand_dims(x.max(axis = 1), axis = 1)
		projx = np.exp(x - rowmax)
		rowsum = np.expand_dims(projx.sum(axis = 1), axis = 1)
		projx = projx/rowsum
		return projx
	elif len(x.shape) == 1:
		b = x.max()
		projx = np.exp(x - b)
		return projx / (projx.sum())
	else:
		print('The shape of x was unexpected')
		return None

#projection onto the l^\infty ball of radius beta
def inf_ball_proj(mu, argdict):
	beta = argdict['infballrad']
	return np.minimum(np.abs(mu), beta) * np.sign(mu)

########################################################################
#        Stochastic Linear Inverse Problems on the Simplex
########################################################################

def stoch_primal_step(x, mu, primal_map, primal_prox, nabf, linopT, argdict):
	#lucky is the set of indices that were picked for the minibatch
	argdict['lucky'] = np.random.choice(argdict['A'].shape[0], argdict['batchsize'], replace=False)
	argdict['lA'] = argdict['A'][argdict['lucky']]
	argdict['ly'] = argdict['y'][argdict['lucky']]
	#This is an adaptive step size with Lp computed only with respect to the sampled rows
	#luckyLp = 0
	#for i in range(argdict['lA'].shape[1]):
	#	luckyLp = np.maximum(luckyLp, argdict['lA'][:,i].sum())
	#luckylam = 1. / (luckyLp + 2.01)
	#u = primal_map(x, argdict) - luckylam * (nabf(x, argdict) + linopT(mu, argdict))
	#this is the ordinary step size
	lam = argdict['primalsize']
	u = primal_map(x, argdict) - lam * (nabf(x, argdict) + linopT(mu, argdict))
	return primal_prox(u, argdict)

#stochastic gradient of KL divergence with linear operator A, \nabla KL(Ax,y)
def stoch_lin_inv_simp_gradf(x, argdict):
	grad = np.dot(argdict['lA'].transpose(), np.log(np.dot(argdict['lA'],x) / argdict['ly']))
	return grad


########################################################################
#            Linear Inverse Problems on the Simplex
########################################################################


#gradient of KL divergence with linear operator A, \nabla KL(Ax,y)
def lin_inv_simp_gradf(x, argdict):
	A = argdict['A']
	y = argdict['y']
	logAxovery = np.log(np.dot(A,x) / y)
	return np.dot(A.transpose(), logAxovery)

#gradient of a vector x in the total variation sense
def lin_inv_simp_linop(x, argdict=None):
	x1 = x[0:-1]
	x2 = x[1:]
	return x2 - x1

#adjoint of the vecgrad operator
def lin_inv_simp_linopT(mu, argdict=None):
	mu1 = np.append(0, mu)
	mu2 = np.append(mu, 0)
	return mu1 - mu2

#the function f from the objective, a kl divergence with linear operator
#f(x) = KL(Ax, y)
def lin_inv_simp_f(x, argdict):
	y = argdict['y']
	A = argdict['A']
	Ax = np.dot(A,x)
	return (Ax * np.log(Ax / y) - Ax + y).sum()

#compute the lagrangian but with the parts containing xsol precomputed
#since the optimality gap is L(x, musol) - L(xsol, mu)
def lin_inv_simp_lagr_x(x, argdict):
	y = argdict['y']
	A = argdict['A']
	Ax = np.dot(A,x)
	kl = (Ax * np.log(Ax / y) - Ax + y).sum()
	return kl + np.dot(x, argdict['linopTmusol'])

#compute the lagrangian but with the parts containing musol precomputed
#since the optimality gap is L(x, musol) - L(xsol, mu)
def lin_inv_simp_lagr_mu(mu, argdict):
	return np.dot(mu, argdict['linopxsol']) + argdict['fxsol']


########################################################################
#            Trend Filtering on the Simplex
########################################################################


#gradient of the sum of KL divergences with linear operators A_i
def trend_filt_gradf(x, argdict):
	y = argdict['y']
	Alist = argdict['Alist']
	grad = np.zeros(x.shape)
	for i in range(x.shape[0]):
		A = Alist[i]
		logAxovery = np.log(np.dot(A,x[i]) / y[i])
		grad[i] = np.dot(A.transpose(), logAxovery)
	return grad

#gradient of the sum of KL divergences with no linear operators
def trend_filt_gradf_id(x, argdict):
	y = argdict['y']
	return np.log(x / y)

#accepts an nxm matrix and computes the rowwise gradient
def trend_filt_linop(x, argdict=None):
	x1 = x[0:-1]
	x2 = x[1:]
	return x2 - x1

#accepts a (n-1)xm matrix and computes the adjoint to the rowwise gradient
def trend_filt_linopT(mu, argdict=None):
	colnum = mu.shape[1]
	mu1 = np.append(np.expand_dims(np.zeros(colnum), axis = 0), mu, axis = 0)
	mu2 = np.append(mu, np.expand_dims(np.zeros(colnum), axis = 0), axis = 0)
	return mu1 - mu2

#the f objective function, a sum of kl divergences with linear operators A_i
def trend_filt_f(x, argdict):
	y = argdict['y']
	Alist = argdict['Alist']
	klsum = 0
	for i in range(x.shape[0]):
		A = Alist[i]
		Ax = np.dot(A, x[i])
		klsum += (Ax * np.log(Ax / y[i]) - Ax + y[i]).sum()
	return klsum

#the f objective function, a sum of kl divergences without linear operators
def trend_filt_f_id(x, argdict):
	y = argdict['y']
	return (x * np.log(x / y) - x + y).sum()

#compute the lagrangian but with the parts containing xsol precomputed
#since the optimality gap is L(x, musol) - L(xsol, mu)
def trend_filt_lagr_x(x, argdict):
	y = argdict['y']
	Alist = argdict['Alist']
	klsum = 0
	for i in range(x.shape[0]):
		A = Alist[i]
		Ax = np.dot(A, x[i])
		klsum += (Ax * np.log(Ax / y[i]) - Ax + y[i]).sum()
	return (argdict['linopTmusol'] * x).sum() + klsum
	
def trend_filt_lagr_x_id(x, argdict):
	y = argdict['y']
	klsum = (x * np.log(x / y) - x + y).sum()
	return (argdict['linopTmusol'] * x).sum() + klsum

#compute the lagrangian but with the parts containing musol precomputed
#since the optimality gap is L(x, musol) - L(xsol, mu)
def trend_filt_lagr_mu(mu, argdict):
	return (argdict['linopxsol'] * mu).sum() + argdict['fxsol']


########################################################################
#            Wasserstein Regularized Inverse Problems
########################################################################

#psi uses the fft to compute the convolution compared to the direct way
def wasserstein_psi(f, argdict):
	n = argdict['dimension']
	padlength = argdict['padlength']
	ghat = argdict['ghat']
	fhat = np.fft.rfft(np.pad(f, (padlength,), 'constant'))
	return np.fft.fftshift(np.real(np.fft.irfft(fhat * ghat)))[padlength:n + padlength]

#the function h in the objective, a sum of logsumexp
def wasserstein_old_h(mu, argdict):
	theta = argdict['theta']
	n = argdict['dimension']
	gamma = argdict['gamma']
	cmat = argdict['costmatrix']
	tau = mu[0:n]
	u = (tau - cmat) / gamma
	umax = u.max(axis = 1)
	expanded_umax = np.expand_dims(umax, axis = 1)
	lse = umax + np.log(np.sum(np.exp(u - expanded_umax), axis = 1))
	return gamma * np.dot(lse, theta)
def wasserstein_h(mu, argdict):
	theta = argdict['theta']
	n = argdict['dimension']
	gamma = argdict['gamma']
	ghat = argdict['ghat']
	tau = mu[0:n]
	tmax = np.max(tau)
	u = np.exp((tau - tmax) / gamma)
	psi = wasserstein_psi(u, argdict)
	lse = (tmax / gamma) + np.log(psi)
	return gamma * np.dot(lse, theta)

#compute the lagrangian but with the parts containing musol precomputed
#since the optimality gap is L(x, musol) - L(xsol, mu)
def wasserstein_lagr_x(x, argdict):
	return np.dot(x, argdict['linopTmusol']) - argdict['hmusol']

#compute the lagrangian but with the parts containing xsol precomputed
#since the optimality gap is L(x, musol) - L(xsol, mu)
def wasserstein_old_lagr_mu(mu, argdict):
	theta = argdict['theta']
	n = argdict['dimension']
	gamma = argdict['gamma']
	cmat = argdict['costmatrix']
	tau = mu[0:n]
	u = (tau - cmat) / gamma
	umax = u.max(axis = 1)
	expanded_umax = np.expand_dims(umax, axis = 1)
	lse = umax + np.log(np.sum(np.exp(u - expanded_umax), axis = 1))
	return np.dot(argdict['linopxsol'], mu) - gamma * np.dot(lse, theta)
def wasserstein_lagr_mu(mu, argdict):
	theta = argdict['theta']
	n = argdict['dimension']
	gamma = argdict['gamma']
	ghat = argdict['ghat']
	tau = mu[0:n]
	tmax = np.max(tau)
	u = np.exp((tau - tmax) / gamma)
	psi = wasserstein_psi(u, argdict)
	lse = (tmax / gamma) + np.log(psi)
	return np.dot(argdict['linopxsol'], mu) - gamma * np.dot(lse, theta)
	
#gradient of logsumexp term from the dual function h
def wasserstein_old_gradh(mu, argdict):
	theta = argdict['theta']
	gamma = argdict['gamma']
	cmat = argdict['costmatrix']
	n = argdict['dimension']
	grad = np.zeros(mu.shape)
	tau = mu[0:n]
	u = (tau - cmat) / gamma
	umax = np.expand_dims(u.max(axis = 1), axis = 1)
	numerator = np.exp(u - umax)
	denominator = np.expand_dims(np.sum(numerator, axis = 1), axis = 1)
	grad[0:n] = np.dot((numerator / denominator).transpose(), theta)
	return grad
def wasserstein_gradh(mu, argdict):
	theta = argdict['theta']
	n = argdict['dimension']
	gamma = argdict['gamma']
	grad = np.zeros(mu.shape)
	tau = mu[0:n]
	tmax = np.max(tau)
	u = np.exp((tau - tmax) / gamma)
	#u = np.exp(tau / gamma)
	denompsi = wasserstein_psi(u, argdict)
	numpsi = wasserstein_psi(theta / denompsi, argdict)
	grad[0:n] = u * numpsi
	return grad

#dual prox
#projects the parts of mu corresponding to the gradient constraint on the infball
def wasserstein_dualprox(mu, argdict):
	beta = argdict['infballrad']
	n = argdict['dimension']
	mu[n:] = np.minimum(np.abs(mu[n:]), beta) * np.sign(mu[n:])
	return mu

#linear operator
def wasserstein_linop(x, argdict=None):
	x1 = x[0:-1]
	x2 = x[1:]
	nabx = x2 - x1
	return np.concatenate([x,nabx])

#adjoint of the linear operator above
def wasserstein_linopT(mu, argdict):
	n = argdict['dimension']
	tau = mu[0:n]
	zeta = mu[n:]
	zeta1 = np.append(0, zeta)
	zeta2 = np.append(zeta, 0)
	return tau + zeta1 - zeta2

#convolution with the bump function
def wasserstein_conv(x, argdict):
	bumpF = argdict['bumpF']
	xF = np.fft.rfft(x)
	x1 = x[0:-1]
	x2 = x[1:]
	nabx = x2 - x1	
	return np.concatenate([np.real(np.fft.fftshift(np.fft.irfft(bumpF * xF))),nabx])

#adjoint of the convolution with the bump function
def wasserstein_convT(mu, argdict):
	n = argdict['dimension']
	bumpFconj = argdict['bumpFconj']
	tau = mu[0:n]
	tauF = np.fft.rfft(tau)
	tau = np.real(np.fft.fftshift(np.fft.irfft(bumpFconj * tauF)))
	zeta = mu[n:]
	zeta1 = np.append(0, zeta)
	zeta2 = np.append(zeta, 0)
	return tau + zeta1 - zeta2

########################################################################
#            Wasserstein Regularized Inverse Problems (IMAGE)
########################################################################

#psi uses the fft to compute the convolution
def wimg_psi(f, argdict):
	n = argdict['dimension']
	padlength = argdict['padlength']
	ghat = argdict['ghat']
	fhat = np.fft.rfft(np.pad(f, (padlength,), 'constant'))
	return np.fft.fftshift(np.real(np.fft.irfft(fhat * ghat)))[padlength:n + padlength]

def wimg_h(mu, argdict):
	theta = argdict['theta']
	n = argdict['dimension']
	gamma = argdict['gamma']
	ghat = argdict['ghat']
	tau = mu[0:n]
	tmax = np.max(tau)
	u = np.exp((tau - tmax) / gamma)
	psi = wimg_psi(u, argdict)
	lse = (tmax / gamma) + np.log(psi)
	return gamma * np.dot(lse, theta)
	
#compute the lagrangian but with the parts containing musol precomputed
#since the optimality gap is L(x, musol) - L(xsol, mu)
def wimg_lagr_x(x, argdict):
	return np.dot(x, argdict['linopTmusol']) - argdict['hmusol']

def wimg_lagr_mu(mu, argdict):
	theta = argdict['theta']
	n = argdict['dimension']
	gamma = argdict['gamma']
	ghat = argdict['ghat']
	tau = mu[0:n]
	tmax = np.max(tau)
	u = np.exp((tau - tmax) / gamma)
	psi = wimg_psi(u, argdict)
	lse = (tmax / gamma) + np.log(psi)
	return np.dot(argdict['linopxsol'], mu) - gamma * np.dot(lse, theta)
	
def wimg_gradh(mu, argdict):
	theta = argdict['theta']
	n = argdict['dimension']
	gamma = argdict['gamma']
	ghat = argdict['ghat']
	grad = np.zeros(mu.shape)
	tau = mu[0:n]
	tmax = np.max(tau)
	u = np.exp((tau - tmax) / gamma)
	denompsi = wimg_psi(u, argdict)
	numpsi = wimg_psi(theta / denompsi, argdict)
	grad[0:n] = u * numpsi
	return grad
	
#dual prox
#projects the parts of mu corresponding to the gradient constraint on the infball
def wimg_dualprox(mu, argdict):
	beta = argdict['infballrad']
	n = argdict['dimension']
	mu[n:] = np.minimum(np.abs(mu[n:]), beta) * np.sign(mu[n:])
	return mu

#linear operator
def wimg_linop(x, argdict=None):
	x1 = x[0:-1]
	x2 = x[1:]
	nabx = x2 - x1
	return np.concatenate([x,nabx])

#adjoint of the linear operator above
def wimg_linopT(mu, argdict):
	n = argdict['dimension']
	tau = mu[0:n]
	zeta = mu[n:]
	zeta1 = np.append(0, zeta)
	zeta2 = np.append(zeta, 0)
	return tau + zeta1 - zeta2

#convolution with the bump function
def wimg_conv(x, argdict):
	bumpF = argdict['bumpF']
	xF = np.fft.rfft(x)
	x1 = x[0:-1]
	x2 = x[1:]
	nabx = x2 - x1	
	return np.concatenate([np.real(np.fft.fftshift(np.fft.irfft(bumpF * xF))),nabx])

#adjoint of the convolution with the bump function
def wimg_convT(mu, argdict):
	n = argdict['dimension']
	bumpFconj = argdict['bumpFconj']
	tau = mu[0:n]
	tauF = np.fft.rfft(tau)
	tau = np.real(np.fft.fftshift(np.fft.irfft(bumpFconj * tauF)))
	zeta = mu[n:]
	zeta1 = np.append(0, zeta)
	zeta2 = np.append(zeta, 0)
	return tau + zeta1 - zeta2


########################################################################
# 			Regularized Wasserstein Barycenter problem
########################################################################


#the function h in the objective, a sum of sums sum of logsumexp
def barycenter_h(mu, argdict):
	n1 = argdict['dimension1']
	n2 = argdict['dimension2']
	theta1 = argdict['theta1']
	theta2 = argdict['theta2']
	epsilon1 = argdict['epsilon1']
	epsilon2 = argdict['epsilon2']
	cmat1 = argdict['costmatrix1']
	cmat2 = argdict['costmatrix2']
	alpha1 = argdict['alpha1']
	alpha2 = argdict['alpha2']
	lam1 = mu[0:n1]
	lam2 = mu[n1:n1 + n2]
	u1 = (lam1 - cmat1) / epsilon1
	u2 = (lam2 - cmat2) / epsilon2
	umax1 = u1.max(axis = 1)
	umax2 = u2.max(axis = 1)
	expanded_umax1 = np.expand_dims(umax1, axis = 1)
	expanded_umax2 = np.expand_dims(umax2, axis = 1)
	lse1 = umax1 + np.log(np.sum(np.exp(u1 - expanded_umax1), axis = 1))
	lse2 = umax2 + np.log(np.sum(np.exp(u2 - expanded_umax2), axis = 1))
	#test
	#u = (lam - cmat) / epsilon
	#umax = u.max(axis=1)
	#expanded_umax = np.expand_dims(umax, axis=1)
	#lse = umax + np.log(np.sum(np.exp(u-expanded_umax), axis=1))
	return alpha1 * epsilon1 * np.dot(lse1, theta1) + alpha2 * epsilon2 * np.dot(lse2, theta2)
	
#compute the lagrangian but with the parts containing musol precomputed
#since the optimality gap is L(x, musol) - L(xsol, mu)
def barycenter_lagr_x(x, argdict):
	return np.dot(x, argdict['linopTmusol']) - argdict['hmusol']

#compute the lagrangian but with the parts containing xsol precomputed
#since the optimality gap is L(x, musol) - L(xsol, mu)
def barycenter_lagr_mu(mu, argdict):
	n1 = argdict['dimension1']
	n2 = argdict['dimension2']
	theta1 = argdict['theta1']
	theta2 = argdict['theta2']
	epsilon1 = argdict['epsilon1']
	epsilon2 = argdict['epsilon2']
	cmat1 = argdict['costmatrix1']
	cmat2 = argdict['costmatrix2']
	alpha1 = argdict['alpha1']
	alpha2 = argdict['alpha2']
	lam1 = mu[0:n1]
	lam2 = mu[n1:n1 + n2]
	u1 = (lam1 - cmat1) / epsilon1
	u2 = (lam2 - cmat2) / epsilon2
	umax1 = u1.max(axis = 1)
	umax2 = u2.max(axis = 1)
	expanded_umax1 = np.expand_dims(umax1, axis = 1)
	expanded_umax2 = np.expand_dims(umax2, axis = 1)
	lse1 = umax1 + np.log(np.sum(np.exp(u1 - expanded_umax1), axis = 1))
	lse2 = umax2 + np.log(np.sum(np.exp(u2 - expanded_umax2), axis = 1))
	return np.dot(argdict['linopxsol'], mu) - (alpha1 * epsilon1 * np.dot(lse1, theta1) + alpha2 * epsilon2 * np.dot(lse2, theta2))
	
#gradient of logsumexp term from the dual function h
def barycenter_gradh(mu, argdict):
	n1 = argdict['dimension1']
	n2 = argdict['dimension2']
	theta1 = argdict['theta1']
	theta2 = argdict['theta2']
	epsilon1 = argdict['epsilon1']
	epsilon2 = argdict['epsilon2']
	cmat1 = argdict['costmatrix1']
	cmat2 = argdict['costmatrix2']
	alpha1 = argdict['alpha1']
	alpha2 = argdict['alpha2']
	grad = np.zeros(mu.shape)
	lam1 = mu[0:n1]
	lam2 = mu[n1:n1 + n2]
	u1 = (lam1 - cmat1) / epsilon1
	u2 = (lam2 - cmat2) / epsilon2
	umax1 = np.expand_dims(u1.max(axis = 1), axis = 1)
	umax2 = np.expand_dims(u2.max(axis = 1), axis = 1)
	numerator1 = np.exp(u1 - umax1)
	numerator2 = np.exp(u2 - umax2)
	denominator1 = np.expand_dims(np.sum(numerator1, axis = 1), axis = 1)
	denominator2 = np.expand_dims(np.sum(numerator2, axis = 1), axis = 1)
	grad[0:n1] = alpha1 * np.dot((numerator1 / denominator1).transpose(), theta1)
	grad[n1:n1 + n2] = alpha2 * np.dot((numerator2 / denominator2).transpose(), theta2)
	return grad
	
#dual prox
#projects the parts of mu corresponding to the gradient constraint on the infball
def barycenter_dualprox(mu, argdict):
	n = argdict['dimension1'] + argdict['dimension2']
	beta = argdict['infballrad']
	mu[n:] = np.minimum(np.abs(mu[n:]), beta) * np.sign(mu[n:])
	return mu	

def barycenter_conv(x, argdict):
	bump1F = argdict['bump1F']
	bump2F = argdict['bump2F']
	xF = np.fft.rfft(x)
	conv1 = np.real(np.fft.fftshift(np.fft.irfft(bump1F * xF)))
	conv2 = np.real(np.fft.fftshift(np.fft.irfft(bump2F * xF)))
	x1 = x[0:-1]
	x2 = x[1:]
	nabx = x2 - x1
	return np.concatenate([conv1, conv2, nabx])
	
def barycenter_convT(mu, argdict):
	n1 = argdict['dimension1']
	n2 = argdict['dimension2']
	bump1Fconj = argdict['bump1Fconj']
	bump2Fconj = argdict['bump2Fconj']
	lam1 = mu[0:n1]
	lam2 = mu[n1:n1 + n2]
	lam1F = np.fft.rfft(lam1)
	lam2F = np.fft.rfft(lam2)
	lam1 = np.real(np.fft.fftshift(np.fft.irfft(bump1Fconj * lam1F)))
	lam2 = np.real(np.fft.fftshift(np.fft.irfft(bump2Fconj * lam2F)))
	v = mu[n1 + n2:]
	v1 = np.append(0, v)
	v2 = np.append(v, 0)
	return lam1 + lam2 + v1 - v2
