import numpy as np
from scipy.special import eval_legendre, roots_legendre    
    
def pl_project_2D(f, L, even=True) :
    """
    Projection of function, assumed to be already evaluated 
    at the Gauss-Legendre scheme points, over the Legendre 
    polynomials.    

    Parameters
    ----------
    f : array_like, shape (N, M)
        function to project.
    L : integer
        truncation order for the harmonic series expansion.
    even : boolean, optional
        should the function assume that f is even?

    Returns
    -------
    f_l : array_like, shape (N, L)
        The projection of f over the legendre polynomials
        for each radial value.

    """
    N, M = np.atleast_2d(f).shape
    cth, weights = roots_legendre(M)
    zeros = lambda f: np.squeeze(np.zeros((N, )))
    project = lambda f, l: f @ (weights * eval_legendre(l, cth))
    norm = (2*np.arange(L)+1)/2
    if even :
        f_l = norm * np.array(
            [project(f, l) if (l%2 == 0) else zeros(f) for l in range(L)]
        ).T
    else : 
        f_l = norm * np.array([project(f, l) for l in range(L)]).T
    return f_l


def pl_eval_2D(f_l, t, der=0) :
    """
    Evaluation of f(r, t) (and its derivatives) from a projection,
    f_l(r, l), of f over the Legendre polynomials.

    Parameters
    ----------
    f_l : array_like, shape (N, L)
        The projection of f over the legendre polynomials.
    t : array_like, shape (N_t, )
        The points on which to evaluate f.
    der : integer in {0, 1, 2}
        The upper derivative order. The default value is 0.
    Returns
    -------
    f : array_like, shape (N, N_t)
        The evaluation of f over t.
    df : array_like, shape (N, N_t), optional
        The evaluation of the derivative f over t.
    d2f : array_like, shape (N, N_t), optional
        The evaluation of the 2nd derivative of f over t.

    """
    assert der in {0, 1, 2} # Check the der input
    
    # f computation
    _, L = np.atleast_2d(f_l).shape
    pl = np.array([eval_legendre(l, t) for l in range(L)])
    f = f_l @ pl
    
    if der != 0 :
        # df computation
        ll = np.arange(L)[:, None]
        dpl = ll * np.roll(pl, 1, axis=0)
        for l in range(1, L):
            dpl[l] += t * dpl[l-1]
        df = f_l @ dpl
        
        if der != 1 :
            # d2f computation
            llp1 = np.where(ll != 0, ll+1, 0)
            d2pl = llp1 * np.roll(dpl, 1, axis=0)
            for l in range(1, L):
                d2pl[l] += t * d2pl[l-1]
            d2f = f_l @ d2pl
            
            return f, df, d2f
        return f, df
    return f