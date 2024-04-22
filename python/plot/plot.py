import matplotlib        as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import numpy             as np
from matplotlib             import rc, ticker
from matplotlib.collections import LineCollection
from pylab                  import cm

from legendre               import pl_eval_2D, pl_project_2D

    
def get_cmap_from_proplot(cmap_name, **kwargs) :
    '''
    Get a colormap defined in the proplot extension. If proplot 
    isn't installed, then return a matplotlib colormap corresponding
    to cmap_name.
    
    Parameters
    ----------
    cmap_name: string
        String corresponding to the colormap name in proplot.
        
    Returns
    -------
    cmap: Colormap instance
        Corresponding colormap.
    '''
    from importlib.util import find_spec
    spec = find_spec('proplot')
    
    if spec is None :  # proplot is not installed
        try : 
            cmap = cm.get_cmap(cmap_name)
        except : 
            stellar_list = ["#fffffe", "#f6cf77", "#bd7a37", "#6a1707", "#1d1d1d"][::-1]
            # fire_list    = ["#fffdfb", "#f7be7a", "#d96644", "#8f3050", "#401631"][::-1]
            cmap = get_continuous_cmap(stellar_list)
        return cmap
    else :             # proplot is installed
        import proplot as pplt
        cmap = pplt.Colormap(cmap_name, **kwargs)
        return cmap

def plot_f_map(
    map_n, f, phi_eff, max_degree,
    angular_res=501, t_deriv=0, levels=100, cmap=cm.Blues, size=16, label=r"$f$",
    show_surfaces=False, n_lines=30, cmap_lines=cm.BuPu, lw=0.5,
    disc=None, disc_color='white', map_ext=None, n_lines_ext=20,
    add_to_fig=None, background_color='white',
) :
    """
    Shows the value of f in the 2D model.

    Parameters
    ----------
    map_n : array_like, shape (N, M)
        2D Mapping.
    f : array_like, shape (N, ) or (N, M)
        Function value on the surface levels or at each point on the mapping.
    phi_eff : array_like, shape (N, )
        Value of the effective potential on each isopotential.
        Serves the colormapping if show_surfaces=True.
    max_degree : integer
        number of harmonics to use for interpolating the mapping.
    angular_res : integer, optional
        angular resolution used to plot the mapping. The default is 501.
    t_deriv : integer, optional
        derivative (with respect to t = cos(theta)) order to plot. Only used
        is len(f.shape) == 2. The default is 0.
    levels : integer, optional
        Number of color levels on the plot. The default is 100.
    cmap : cm.cmap instance, optional
        Colormap for the plot. The default is cm.Blues.
    size : integer, optional
        Fontsize. The default is 16.
    label : string, optional
        Name of the f variable. The default is r"$f$"
    show_surfaces : boolean, optional
        Show the isopotentials on the left side if set to True.
        The default is False.
    n_lines : integer, optional
        Number of equipotentials on the plot. The default is 50.
    cmap_lines : cm.cmap instance, optional
        Colormap used for the isopotential plot. 
        The default is cm.BuPu.
    disc : array_like, shape (Nd, ), optional
        Indices of discontinuities to plot. The default is None.
    disc_color : string, optional
        Color used to display the discontinuities. The default is 'white'.
    map_ext : array_like, shape (Ne, M), optional
        Used to show the external mapping, if given.
    n_lines_ext : integer, optional
        Number of level surfaces in the external mapping. The default is 20.
    add_to_fig : fig object, optional
        If given, the figure on which the plot should be added. 
        The default is None.
    background_color : string, optional
        Optional color for the plot background. The default is 'white'.

    Returns
    -------
    None.

    """
    
    # Angular interpolation
    N, M = map_n.shape
    cth_res = np.linspace(-1, 1, angular_res)
    sth_res = np.sqrt(1-cth_res**2)
    map_l   = pl_project_2D(map_n, max_degree)
    map_res = pl_eval_2D(map_l, cth_res)
    
    # 2D density
    if len(f.shape) == 1 :
        f2D = np.tile(f, angular_res).reshape((angular_res, -1)).T
    else : 
        f_l = pl_project_2D(f, max_degree, even=False)
        f2D = np.atleast_3d(np.array(pl_eval_2D(f_l, cth_res, der=t_deriv)).T).T[-1]
    Nf = f2D.shape[0]
        
    # Text formating 
    rc('text', usetex=True)
    rc('xtick', labelsize=size)
    rc('ytick', labelsize=size)
    rc('axes', facecolor=background_color)
    
    # Init figure
    norm = None
    if sum((1.0-np.array(cm.get_cmap(cmap)(0.5)[:3]))**2) < 1e-2 : # ~ Test if the cmap is divergent
        norm = mcl.CenteredNorm()
    cbar_width = 0.1
    if add_to_fig is None : 
        margin = 0.05
        x_scale = 2 * margin + (map_res[-1]*sth_res).max() + 4 * cbar_width
        y_scale = 2 * margin + (map_res[-1]*cth_res).max()
        factor = min(18/x_scale, 9.5/y_scale)
        fig, ax = plt.subplots(figsize=(x_scale * factor, y_scale * factor), frameon=False)
    else : 
        fig, ax = add_to_fig
    
    # Right side
    csr = ax.contourf(
        map_res[N-Nf:]*sth_res, map_res[N-Nf:]*cth_res, f2D, 
        cmap=cmap, norm=norm, levels=levels
    )
    for c in csr.collections:
        c.set_edgecolor("face")
    if disc is not None :
        for i in disc :
            plt.plot(map_res[i]*sth_res, map_res[i]*cth_res, color=disc_color, lw=lw)
    plt.plot(map_res[-1]*sth_res, map_res[-1]*cth_res, 'k-', lw=lw)
    cbr = fig.colorbar(csr, pad=0.7*cbar_width, fraction=cbar_width, shrink=0.85, aspect=25)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbr.locator = tick_locator
    cbr.update_ticks()
    cbr.ax.set_title(label, y=1.03, fontsize=size+3)
    
    # Left side
    if show_surfaces :
        ls = LineCollection(
            [np.column_stack([x, y]) for x, y in zip(
                -map_res[::-N//n_lines]*sth_res, 
                 map_res[::-N//n_lines]*cth_res
            )], 
            cmap=cmap_lines, 
            linewidths=lw
        )
        ls.set_array(phi_eff[::-N//n_lines])
        ax.add_collection(ls)
        cbl = fig.colorbar(
            ls, location='left', pad=cbar_width, fraction=cbar_width, shrink=0.85, aspect=25
        )
        cbl.locator = tick_locator
        cbl.update_ticks()
        cbl.ax.set_title(
            r"$\phi_\mathrm{eff} \times \left(GM/R_\mathrm{eq}\right)^{-1}$", 
            y=1.03, fontsize=size+3
        )
    else : 
        csl = ax.contourf(
            -map_res[N-Nf:]*sth_res, map_res[N-Nf:]*cth_res, f2D, 
            cmap=cmap, norm=norm, levels=levels
        )
        for c in csl.collections:
            c.set_edgecolor("face")
        if disc is not None :
            for i in disc :
                plt.plot(-map_res[i]*sth_res, map_res[i]*cth_res, 'w-', lw=lw)
        plt.plot(-map_res[-1]*sth_res, map_res[-1]*cth_res, 'k-', lw=lw)
        
    # External mapping
    if map_ext is not None : 
        Ne, _ = map_ext.shape
        map_ext_l   = pl_project_2D(map_ext, max_degree)
        map_ext_res = pl_eval_2D(map_ext_l, np.linspace(-1, 1, angular_res))
        for ri in map_ext_res[::-Ne//n_lines_ext] : 
            plt.plot( ri*sth_res, ri*cth_res, lw=lw/2, ls='-', color='grey')
            plt.plot(-ri*sth_res, ri*cth_res, lw=lw/2, ls='-', color='grey')
    
    # Show figure
    plt.axis('equal')
    plt.xlabel('$s/R_\mathrm{eq}$', fontsize=size+3)
    plt.ylabel('$z/R_\mathrm{eq}$', fontsize=size+3)
    plt.xlim((-1.0, 1.0))
    fig.tight_layout()
    plt.show()
    
def plot_scalar(
    r, f, L, potential=False, J_out=501, levels=100, cmap=cm.Blues, lw=0.5, div=False, 
    disc=None, disc_color='white', size=16, label=r"$f$"
) :
    """
    Shows the value of f in the 2D model.

    Parameters
    ----------
    r : array_like, shape (N, M)
        2D Mapping.
    f : array_like, shape (N, ) or (N, M)
        Function value on the surface levels or at each point on the mapping.
    L : integer
        number of harmonics to use for interpolating the mapping.
    potential : boolean, optional
        Whether the field is a potential to plot the contour or not.
    J_out : integer, optional
        angular resolution used to plot the mapping. The default is 501.
    levels : integer, optional
        Number of color levels on the plot. The default is 100.
    cmap : cm.cmap instance, optional
        Colormap for the plot. The default is cm.Blues.
    div : boolean, optional
        Specifies if the cmap is divergent or not.
    disc : array_like, shape (Nd, ), optional
        Indices of discontinuities to plot. The default is None.
    disc_color : string, optional
        Color used to display the discontinuities. The default is 'white'.
    size : integer, optional
        Fontsize. The default is 16.
    label : string, optional
        Name of the f variable. The default is r"$f$"

    Returns
    -------
    None.

    """
    
    # Angular interpolation
    I, J = r.shape
    c_out = np.linspace(-1, 1, J_out)
    s_out = np.sqrt(1-c_out**2)
    r_l   = pl_project_2D(r, L)
    r_out = pl_eval_2D(r_l, c_out)
    
    # 2D density
    if len(f.shape) == 1 :
        f_out = np.tile(f, J_out).reshape((J_out, -1)).T
    else : 
        f_l = pl_project_2D(f, L, even=False)
        f_out = np.atleast_3d(np.array(pl_eval_2D(f_l, c_out)).T).T[-1]
    If = f_out.shape[0]
    
    # Both sides
    x_out, y_out = r_out*s_out, r_out*c_out
    x_out = np.hstack((x_out[:, ::-1], -x_out, x_out[:, -1, None]))
    y_out = np.hstack((y_out[:, ::-1],  y_out, y_out[:, -1, None]))
    f_out = np.hstack((f_out[:, ::-1],  f_out, f_out[:, -1, None]))
        
    # Text formating 
    rc('text' , usetex=True)
    rc('xtick', labelsize=size)
    rc('ytick', labelsize=size)
    
    # Init figure
    norm = None
    if div : norm = mcl.CenteredNorm()
    cbar_width = 0.1
    margin = 0.05
    x_scale = 2 * margin + x_out[-1].max() + 4 * cbar_width
    y_scale = 2 * margin + y_out[-1].max()
    factor = min(18/x_scale, 9.5/y_scale)
    fig, ax = plt.subplots(figsize=(x_scale * factor, y_scale * factor), frameon=False)
    
    # Plot contour
    plot_function = ax.contour if potential is True else ax.contourf
    cs = plot_function(
        x_out[I-If:], y_out[I-If:], f_out, 
        cmap=cmap, norm=norm, levels=levels, 
        linewidths=lw
    )
    for c in cs.collections:
        c.set_edgecolor("face")
    if disc is not None :
        for i in disc :
            plt.plot(x_out[i], y_out[i], color=disc_color, lw=lw)
    plt.plot(x_out[-1], y_out[-1], 'k-', lw=lw)
    cb = fig.colorbar(cs, pad=0.7*cbar_width, fraction=cbar_width, shrink=0.85, aspect=25)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()
    cb.ax.set_title(label, y=1.03, fontsize=size+3)
    
    # Show figure
    plt.axis('equal')
    plt.xlabel('$s/R_\mathrm{eq}$', fontsize=size+3)
    plt.ylabel('$z/R_\mathrm{eq}$', fontsize=size+3)
    plt.xlim((-1.0, 1.0))
    fig.tight_layout()
    plt.show()