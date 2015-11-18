from gpnarx import *
import pylab as pb

num_inducing = 40

# Create toy data
N1 = 140
X1 = np.random.rand(N1, 1) * 8*np.pi
X1.sort(0)
Y1 = np.sin(X1) + np.random.randn(*X1.shape) * 0.02
#U1 = X1%(2*np.pi)

# Train a standard model
m1 = GPy.models.SparseGPRegression(X1,Y1, num_inducing = num_inducing)
m1.optimize('bfgs', max_iters=100)
m1.plot()

# Create transformed data (autoregressive dataset)
ws=10 # Windowsize
xx,yy = transformTimeSeriesToSeq(Y1, ws)

#uu,tmp = transformTimeSeriesToSeq(U1, ws)
# Test the above: np.sin(uu) - xx

#uu = yy**2 -2*yy + 5 + np.random.randn(*yy.shape) * 0.005
U1 = Y1**2 -2*Y1 + 5 + np.random.randn(*Y1.shape) * 0.005
uu,tmp = transformTimeSeriesToSeq(U1, ws)

Xtr = xx[0:50,:]
Xts = xx[50:,:]
Ytr = yy[0:50,:]
Yts = yy[50:,:]
Utr = uu[0:50,:]
Uts = uu[50:,:]


# Train regression model
m = GPy.models.SparseGPRegression(np.hstack((Xtr,Utr)),Ytr, num_inducing = num_inducing)
m.optimize('bfgs', max_iters=1000, messages=True)
print m

# Initial window to kick-off free simulation
x_start = Xts[0,:][:,None].T

# Free simulation
ygp, varygp = gp_narx(m, x_start, Yts.shape[0], Uts, ws)
pb.figure()
pb.plot(Yts, 'x-')
pb.plot(ygp, 'ro-')
pb.legend(('True','Pred'))
pb.title('NARX-full')

Xrand = np.random.randn(*Xtr.shape)
mrandx = GPy.models.SparseGPRegression(np.hstack((Xrand,Utr)),Ytr, num_inducing = num_inducing)
mrandx.optimize('bfgs', max_iters=1000, messages=True)
print mrandx

# Free simulation
ygp, varygp = gp_narx(mrandx, x_start, Yts.shape[0], Uts, ws)
pb.figure()
pb.plot(Yts, 'x-')
pb.plot(ygp, 'ro-')
pb.legend(('True','Pred'))
pb.title('NARX-RAND_X')


Urand = np.random.randn(*Utr.shape)
mrandu = GPy.models.SparseGPRegression(np.hstack((Xtr,Urand)),Ytr, num_inducing = num_inducing)
mrandu.optimize('bfgs', max_iters=1000, messages=True)
print mrandu

# Free simulation
ygp, varygp = gp_narx(mrandu, x_start, Yts.shape[0], Uts, ws)
pb.figure()
pb.plot(Yts, 'x-')
pb.plot(ygp, 'ro-')
pb.legend(('True','Pred'))
pb.title('NARX-RAND_U')


mrandxu = GPy.models.SparseGPRegression(np.hstack((Xrand,Urand)),Ytr, num_inducing = num_inducing)
mrandxu.optimize('bfgs', max_iters=1000, messages=True)
print mrandxu

# Free simulation
ygp, varygp = gp_narx(mrandxu, x_start, Yts.shape[0], Uts, ws)
pb.figure()
pb.plot(Yts, 'x-')
pb.plot(ygp, 'ro-')
pb.legend(('True','Pred'))
pb.title('NARX-RAND_X_U')