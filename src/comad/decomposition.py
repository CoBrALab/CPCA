import numpy as np
from sklearn.utils import check_random_state

from .utils import closed_form
from .eval import eval_comad


class CoMaD:
    """CoMaD core class.

    When calling self.comad_modeling(X), where X are some target timeseries, the following 
    linear model is generated:
    X = W_net@C_net^T + W_comad@C_comad^T + e

    Cleaned timeseries then correspond to the subspace of network components, with
    or without the residuals added, i.e. X = W_net@C_net^T (+ e), and can be generated
    using self.clean().

    Parameters
    ----------
    C_prior : 2D array of shape n_voxels X n_prior, default=None
        Set of prior components.

    N_comad : int, default=0
        Number of CoMaD components to derive.

    aggressive : bool, default=True
        Whether to apply the regression in the temporal domain for a CoMaD correction 
        applied along both space and time axes.

    sequential_decomposition : bool, default=False
        Whether to derive CoMaD components sequentially as opposed to all at once.
        By doing so, each component is derived by maximizing variance explained
        in decreasing order, akin to PCA. The decomposition is then deterministic,
        whereas the parallel decomposition is not.

    compute_residuals : bool, default=False
        Whether to also output the residuals from the CoMaD model.

    gen_report : bool, default=False
        Whether to generate the CoMaD fitting report.

    optimize_N : bool, default=False
        Whether to carry an automated dimensionality estimation for CoMaD, up to 
        a maximal dimensionality defined by N_comad.

    min_prior_sim : float, default=None
        Parameter for automated dimensionality estimation (when optimize_N=True).
        Set a convergence threshold between 0 and 1.0 for the similarity between
        the priors and the associated network maps derived from the CoMaD model. 

    Dc_W_thresh : float, default=None
        Parameter for automated dimensionality estimation (when optimize_N=True).
        Set a convergence threshold for the cosine distance of the resulting network timecourses
        after consecutive increments in CoMaD dimensionality.

    Dc_C_thresh : float, default=None
        Parameter for automated dimensionality estimation (when optimize_N=True).
        Set a convergence threshold for the cosine distance of the resulting network maps
        after consecutive increments in CoMaD dimensionality.

    c_init : 2D array of shape n_voxels X N_comad, default=None
        It is possible to provide a initialization of the CoMaD components.

    tol : float, default=1e-10
        Tolerance threshold for CoMaD convergence.

    verbose : int, default=1
        Verbosity level.

    Attributes
    ----------
    C_net : 2D array of shape n_voxels X n_prior
        Maps of the network components.

    W_net : 2D array of shape n_frames X n_prior
        Timecourses of the network components.

    C_comad : 2D array of shape n_voxels X N_comad
        Maps of the CoMaD components.

    W_comad : 2D array of shape n_frames X N_comad
        Timecourses of the CoMaD components.

    residuals : 2D array of shape n_frames X n_voxels
        Residuals of the linear model, stored only if compute_residuals=True

    n_optim_idx : int
        The optimal CoMaD dimensionality estimated if optimize_N=True

    fig_list : list of matplotlib figures
        Figures generated with gen_report=True.        

    """

    def __init__(self, 
                C_prior=None, N_comad=0,
                aggressive=True, sequential_decomposition=False, compute_residuals=False, 
                gen_report=False, optimize_N=False, min_prior_sim=None, Dc_W_thresh=None, Dc_C_thresh=None,
                c_init=None, tol=1e-10, verbose=1,
                ):
        # set all parameters for the CoMaD decomposition
        self.C_prior = C_prior
        self.N_comad = N_comad
        self.aggressive=aggressive
        self.sequential_decomposition=sequential_decomposition
        self.compute_residuals=compute_residuals 
        self.gen_report=gen_report
        self.optimize_N=optimize_N
        self.min_prior_sim=min_prior_sim
        self.Dc_W_thresh=Dc_W_thresh
        self.Dc_C_thresh=Dc_C_thresh
        self.c_init=c_init
        self.tol=tol
        self.verbose=verbose

    def comad_modeling(self, X):
        """
        Fit the CoMaD model onto an input timeseries matrix.

        Parameters
        ----------
        X : 2D array of shape n_frames X n_voxels
            Input timeseries matrix.
        """

        C_net,W_net,C_comad,W_comad,res,n_optim_idx,fig_list = comad_modeling_(
            X,C_prior=self.C_prior,N_comad=self.N_comad,
            aggressive=self.aggressive, sequential_decomposition=self.sequential_decomposition, compute_residuals=self.compute_residuals, 
            gen_report=self.gen_report, optimize_N=self.optimize_N, min_prior_sim=self.min_prior_sim, Dc_W_thresh=self.Dc_W_thresh, Dc_C_thresh=self.Dc_C_thresh,
            c_init=self.c_init, tol=self.tol, verbose=self.verbose,
            )
        
        self.C_net=C_net
        self.W_net=W_net
        self.C_comad=C_comad
        self.W_comad=W_comad
        if self.compute_residuals:
            self.residuals=res
        if self.optimize_N:
            self.n_optim_idx=n_optim_idx
        if self.gen_report:
            self.fig_list=fig_list
        return self
    
    def clean(self, include_residuals=False):
        if not hasattr(self, "C_net"):
            raise ValueError("Must carry comad_modeling() before clean().")
        # reconstruct the timeseries without the CoMaD components
        X_clean = self.W_net.dot(self.C_net.T)
        if include_residuals:
            if not self.compute_residuals:
                raise ValueError("compute_residuals must be selected during comad_modeling()")
            X_clean+=self.residuals

        return X_clean


def RMS_norm(arr): # RMS normalization along first axis
    return arr/np.sqrt((arr ** 2).mean(axis=0))


def spatial_comad(X, q=1, W_prior=None, c_init=None, tol=1e-6, max_iter=200, verbose=1):
    '''
    Derives spatially orthogonal complementary components
    '''
    # X: time by voxel matrix
    # c_init: can specify an voxel by component number matrix for initiating weights

    if c_init is None:
        random_state = check_random_state(None)
        c_init = random_state.normal(
            size=(X.shape[1], q))
        
    if W_prior is None:
        W_prior = np.zeros([X.shape[0], 0])
    W_prior = W_prior.copy() # copy to avoid editing outside the function
    W_prior = RMS_norm(W_prior)
    
    Cs = c_init.copy()
    Cs = RMS_norm(Cs)

    for i in range(max_iter):
        Cs_prev = Cs
        
        Ws = closed_form(Cs, X.T).T

        C_ = closed_form(np.concatenate((Ws, W_prior), axis=1), X).T
        Cs = C_[:,:q]
        Cs = RMS_norm(Cs)
        
        ##### evaluate convergence
        lim = np.abs(np.abs((Cs * Cs_prev).mean(axis=0)) - 1).mean()
        if verbose > 2:
            print('lim:'+str(lim))
        if lim < tol:
            if verbose > 1:
                print(str(i)+' iterations to converge.')
            break
        if i == max_iter-1:
            if verbose > 0:
                print(
                    'Convergence failed. Consider increasing max_iter or decreasing tol.')                
    
    return Cs,Ws,C_


def comad_modeling_(
        X, C_prior=None, N_comad=0,
        aggressive=True, sequential_decomposition=False, compute_residuals=False, 
        gen_report=False, optimize_N=False, min_prior_sim=None, Dc_W_thresh=None, Dc_C_thresh=None, # manage automated N estimation
        c_init=None, tol=1e-10, verbose=1,
        ):
    
    X = X.copy()

    if C_prior is None:
        C_prior = np.zeros([X.shape[1], 0])
    C_prior = C_prior.copy()
    n_prior = C_prior.shape[1]
    W_prior = closed_form(C_prior, X.T).T

    if (optimize_N or gen_report) and not sequential_decomposition:
        raise ValueError("With optimize_N or gen_report, must also use sequential_decomposition=True to derive components in decreasing order of variance explained.")

    if gen_report or optimize_N:
        if n_prior==0 or N_comad<2:
            raise ValueError("At least one prior and 2 CoMaD components are required.")

    '''
    Derivation of CoMaD spatial components
    '''
    if sequential_decomposition:
        q=1 # one component is derived at a time to have deterministic outputs

        Cs = np.zeros([X.shape[1], 0])
        X_ = X.copy()
        for i in range(N_comad):
            if verbose>1:
                print(i)
            cs,ws,_ = spatial_comad(X_, q=q, W_prior=W_prior, c_init=c_init, tol=tol, verbose=verbose)
            # regress the component out of X so that the next component is computed on an orthogonal subspace
            ws = closed_form(cs, X_.T).T
            X_ = X_ - ws.dot(cs.T)
            Cs = np.concatenate((Cs, cs),axis=1) # append the new component
    else:
        if N_comad>0:
            Cs,_,_ = spatial_comad(X, q=N_comad, W_prior=W_prior, c_init=c_init, tol=tol, verbose=verbose)
        else:
            Cs = np.zeros([X.shape[1], 0])
    
    if gen_report or optimize_N:
        '''
        Generation of QC report and automated dimensionality estimation
        '''

        # generate the fitting report
        n_optim_idx, fig_list = eval_comad(X, C_prior, Cs, min_prior_sim=min_prior_sim, Dc_W_thresh=Dc_W_thresh, Dc_C_thresh=Dc_C_thresh, gen_report=gen_report)

        if not gen_report:
            fig_list = None # don't hold the figures in memory

        if optimize_N:
            if n_optim_idx is None:
                # if no threshold was applied, or min_prior_sim was not met, all components are selected
                Cs = Cs
            else:
                Cs = Cs[:,:n_optim_idx]
    else:
        n_optim_idx=None
        fig_list=None

    '''
    Compute the final linear model
    '''

    W_comad = closed_form(Cs, X.T).T

    if aggressive:
        # components are used as temporal regressors, Wnet is orthogonal from those regressors
        W_net = W_prior - W_comad.dot(closed_form(W_comad,W_prior))
    else: # the prior timecourse is unchanged
        W_net = W_prior

    # normalization in temporal domain so that the amplitude is in the spatial domain
    W_net = RMS_norm(W_net)
    W_comad = RMS_norm(W_comad) 

    W_ = np.concatenate((W_net, W_comad),axis=1)
    C_ = closed_form(W_, X).T

    C_net = C_[:,:n_prior]
    C_comad = C_[:,n_prior:]

    if compute_residuals:
        res = X - W_.dot(C_.T)
    else:
        res = None

    return C_net,W_net,C_comad,W_comad,res,n_optim_idx,fig_list
