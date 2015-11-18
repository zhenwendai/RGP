from gpnarx import *

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
ws=15 # Windowsize
xx,yy = transformTimeSeriesToSeq(Y1, ws)

#uu,tmp = transformTimeSeriesToSeq(U1, ws)
# Test the above: np.sin(uu) - xx

uu = yy**2 -2*yy + 5

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