import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from matplotlib import collections as mc
import plotly.figure_factory as ff
from opt_einsum import contract

def plot_2d(data):
    fig, ax = plt.subplots()
    ax.scatter(data[:,0], data[:,1])
    fig.show()

def plot_3d(data):
    fig = go.Figure(data=[go.Scatter3d(x = data[:,0],
                                    y = data[:,1],
                                    z = data[:,2],
                                    mode='markers')])
    fig.update_traces(marker_size = 3)
    fig.update_layout(scene_aspectmode='data')
    fig.show()

def plot_functions_2d(u, data, rows, cols, cap = -1):
    if cap < 0:
        cap = max(u[:5].max(), -u[:5].min())
    fig, ax = plt.subplots(rows, cols, figsize = (25,18))
    for i in range(rows):
        for j in range(cols):
            ax[i,j].set_aspect('equal')
            ax[i,j].scatter(data[:,0], data[:,1], c = u[cols*i+j], vmin = -cap, vmax = cap)
            # ax.scatter(end_points[:,0], end_points[:,1]) , c = u[K]
            # ax[i,j].set_title(lam[cols*i+j].round(4))

def plot_function_3d(f, data, cap):
    fig = go.Figure(data=[go.Scatter3d(x = data[:,0],
                                    y = data[:,1],
                                    z = data[:,2],
                                    mode='markers',
                                    marker=dict(
                                            size=6,
                                            color=f,                # set color to an array/list of desired values
                                            colorscale='PRGn',   # choose a colorscale
                                            opacity=0.8
                                        ))])
    fig.update_layout(scene_aspectmode='data',
                      width = 800,
                      height = 600)
    fig.update_coloraxes(cmax = cap,
                         cmin = -cap,
                         cmid = 0)
    fig.show()

def vector_field_coordinates(v, u, D, G1_vf, data, parameters):
    n0 = parameters['n0']
    return u.T @ (G1_vf @ v).reshape(n0,n0) @ (u*D @ data)

def vector_field_vis(v, u, D, data, G1_vf, parameters):
    n0 = parameters['n0']
    base_points = u.T @ (u * D) @ data
    end_points = base_points + 5 * u.T @ (G1_vf @ v).reshape(n0,n0) @ (u @ data)
    return base_points, end_points

def plot_1forms_2d(data, U, L, u, G1_vf, D, rows, cols, parameters):
    fig, ax = plt.subplots(rows, cols, figsize = (25,18))
    for i in range(rows):
        for j in range(cols):
            base_points, end_points = vector_field_vis(1*U[:,cols*i+j], u, D, data, G1_vf, parameters)
            ax[i,j].set_aspect('equal')
            ax[i,j].scatter(base_points[:,0], base_points[:,1], c = 'pink')
            ax[i,j].set_title(L[cols*i+j].round(4))
            lc = mc.LineCollection(np.stack((base_points, end_points), axis = 1), color = 'k')
            ax[i,j].add_collection(lc)

def plot_quiver_plain(v, dg_class):

    data = dg_class.data
    quiver = dg_class.vector_field_coords(v)
    g = np.linalg.norm(quiver, axis=-1)
    
    fig = ff.create_quiver(data[:,0], data[:,1], quiver[:,0], quiver[:,1],
                        scale=1,
                        arrow_scale=0.4,
                        line_width=3,
                        marker=dict(color='black'))
    # marker=dict(color=[u.T @ g1])
    fig.add_trace(go.Scatter(x = data[:,0], 
                             y = data[:,1], 
                             mode= 'markers', 
                             marker=dict(size = 7, 
                                         color=g,
                                         colorscale=["gray", "black"],
                                         cmax = 0.1*g.max(),
                                         cmin = g.min())))

    fig.update_layout(width=800, 
                    height=500, 
                    showlegend=False, 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    xaxis_visible=False, 
                    yaxis_visible=False,
                    xaxis_range = [data[:,0].min() - 0.05, data[:,0].max() + 0.05],
                    yaxis_range = [data[:,1].min() - 0.05, data[:,1].max() + 0.05])
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    return fig

def plot_quiver_plain_3d(v, theta_x, theta_z, dg_class):

    Rz = np.array([[np.cos(theta_z),np.sin(theta_z),0],[-np.sin(theta_z),np.cos(theta_z),0],[0,0,1]])
    Rx = np.array([[1,0,0],[0,np.cos(theta_x),np.sin(theta_x)]])

    rotated_data = dg_class.data @ Rz.T @ Rx.T
    quiver = dg_class.vector_field_coords(v) @ Rz.T @ Rx.T
    g = np.linalg.norm(quiver, axis=-1)
    
    fig = ff.create_quiver(rotated_data[:,0], rotated_data[:,1], quiver[:,0], quiver[:,1],
                        scale=1,
                        arrow_scale=0.4,
                        line_width=3,
                        marker=dict(color='black'))
    # marker=dict(color=[u.T @ g1])
    fig.add_trace(go.Scatter(x = rotated_data[:,0], 
                             y = rotated_data[:,1], 
                             mode= 'markers', 
                             marker=dict(size = 7, 
                                         color=g,
                                         colorscale=["gray", "black"],
                                         cmax = 0.1*g.max(),
                                         cmin = g.min())))

    fig.update_layout(width=800, 
                    height=500, 
                    showlegend=False, 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    xaxis_visible=False, 
                    yaxis_visible=False,
                    xaxis_range = [rotated_data[:,0].min() - 0.1, rotated_data[:,0].max() + 0.1],
                    yaxis_range = [rotated_data[:,1].min() - 0.1, rotated_data[:,1].max() + 0.1])
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    return fig

def plot_quiver_plain_tight(v, dg_class):

    data = dg_class.data
    quiver = dg_class.vector_field_coords(v)
    g = np.linalg.norm(quiver, axis=-1)
    
    fig = ff.create_quiver(data[:,0], data[:,1], quiver[:,0], quiver[:,1],
                        scale=1,
                        arrow_scale=0.4,
                        line_width=3,
                        marker=dict(color='black'))
    # marker=dict(color=[u.T @ g1])
    fig.add_trace(go.Scatter(x = data[:,0], 
                             y = data[:,1], 
                             mode= 'markers', 
                             marker=dict(size = 7, 
                                         color=g,
                                         colorscale=["gray", "black"],
                                         cmax = 0.1*g.max(),
                                         cmin = g.min())))

    fig.update_layout(width=500, 
                    height=500, 
                    showlegend=False, 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    xaxis_visible=False, 
                    yaxis_visible=False,
                    xaxis_range = [data[:,0].min() - 0.05, data[:,0].max() + 0.05],
                    yaxis_range = [data[:,1].min() - 0.05, data[:,1].max() + 0.05])
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    return fig

def plot_scatter(data, c, m):

    fig = px.scatter(x = data[:,0], 
                     y = data[:,1], 
                     color=c,
                     color_continuous_scale=["red", "black", "lightgreen"],
                     color_continuous_midpoint=0)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x = data[:,0], 
                             y = data[:,1], 
                             mode= 'markers', 
                             marker=dict(size = 9,
                                         cmid = 0,
                                         cmin = -m,
                                         cmax = m,
                                         color=c,
                                        colorscale=["#166dde", "lightgray", "#e32636"])))

    fig.update_layout(width=800, 
                    height=500, 
                    showlegend=False, 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    xaxis_visible=False, 
                    yaxis_visible=False,
                    xaxis_range = [data[:,0].min() - 0.05, data[:,0].max() + 0.05],
                    yaxis_range = [data[:,1].min() - 0.05, data[:,1].max() + 0.05])
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    return fig

def plot_scatter_tight(data, c, m):

    fig = px.scatter(x = data[:,0], 
                     y = data[:,1], 
                     color=c,
                     color_continuous_scale=["red", "black", "lightgreen"],
                     color_continuous_midpoint=0)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x = data[:,0], 
                             y = data[:,1], 
                             mode= 'markers', 
                             marker=dict(size = 9,
                                         cmid = 0,
                                         cmin = -m,
                                         cmax = m,
                                         color=c,
                                        colorscale=["#166dde", "lightgray", "#e32636"])))

    fig.update_layout(width=500, 
                    height=500, 
                    showlegend=False, 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    xaxis_visible=False, 
                    yaxis_visible=False,
                    xaxis_range = [data[:,0].min() - 0.05, data[:,0].max() + 0.05],
                    yaxis_range = [data[:,1].min() - 0.05, data[:,1].max() + 0.05])
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    return fig


def plot_2_form_plain(v, dg_class):

    data = dg_class.data
    
    n1,n2 = dg_class.parameters['n1'], dg_class.parameters['n2']

    v1 = v.reshape(n1,n2,n2).mean(axis=2).flatten()
    v2 = v.reshape(n1,n2,n2).mean(axis=1).flatten()

    v1_vecs = vector_field_coordinates(v1, dg_class.u(), dg_class.D(), dg_class.G1_VF(), dg_class.data, dg_class.parameters)
    v2_vecs = vector_field_coordinates(v2, dg_class.u(), dg_class.D(), dg_class.G1_VF(), dg_class.data, dg_class.parameters)

    v1_lengths = np.linalg.norm(v1_vecs, axis=1)
    v2_lengths = np.linalg.norm(v2_vecs, axis=1)

    metric = dg_class.u().T @ contract('ijk,i,j',dg_class.g2(),v,v)
    areas = np.sqrt(np.absolute(metric))
    v1_vecs *= (areas/v1_lengths).reshape(-1,1)
    v2_vecs *= (areas/v2_lengths).reshape(-1,1)

    end_points_1 = data + v1_vecs
    end_points_2 = data + v2_vecs

    all_tuples_x = [[a[0], a[1], a[2], None] for a in zip(end_points_1[:,0], data[:,0], end_points_2[:,0])]
    all_tuples_y = [[a[0], a[1], a[2], None] for a in zip(end_points_1[:,1], data[:,1], end_points_2[:,1])]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[p for l in all_tuples_x for p in l],
                            y=[p for l in all_tuples_y for p in l],
                            fill="toself",
                            fillcolor='gray',
                            marker=dict(size = 7, 
                                         color='black',
                                         colorscale=["gray", "black"])))
    fig.add_trace(go.Scatter(x = data[:,0], 
                             y = data[:,1], 
                             mode= 'markers', 
                             marker=dict(size = 7, 
                                         color=metric,
                                         colorscale=["gray", "black"],
                                         cmax = 0.1*metric.max(),
                                         cmin = metric.min())))
    fig.update_layout(width=800, 
                    height=500, 
                    showlegend=False, 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    xaxis_visible=False, 
                    yaxis_visible=False,
                    xaxis_range = [data[:,0].min() - 0.05, data[:,0].max() + 0.05],
                    yaxis_range = [data[:,1].min() - 0.05, data[:,1].max() + 0.05])
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    # fig.show()
    return fig


def plot_2_form_plain_3d(v, theta_x, theta_z, dg_class):

    Rz = np.array([[np.cos(theta_z),np.sin(theta_z),0],[-np.sin(theta_z),np.cos(theta_z),0],[0,0,1]])
    Rx = np.array([[1,0,0],[0,np.cos(theta_x),np.sin(theta_x)]])

    rotated_data = dg_class.data @ Rz.T @ Rx.T
    
    n1,n2 = dg_class.parameters['n1'], dg_class.parameters['n2']

    v1 = v.reshape(n1,n2,n2).mean(axis=2).flatten()
    v2 = v.reshape(n1,n2,n2).mean(axis=1).flatten()

    v1_vecs = vector_field_coordinates(v1, dg_class.u(), dg_class.D(), dg_class.G1_VF(), dg_class.data, dg_class.parameters)
    v2_vecs = vector_field_coordinates(v2, dg_class.u(), dg_class.D(), dg_class.G1_VF(), dg_class.data, dg_class.parameters)

    v1_lengths = np.linalg.norm(v1_vecs, axis=1)
    v2_lengths = np.linalg.norm(v2_vecs, axis=1)

    metric = dg_class.u().T @ contract('ijk,i,j',dg_class.g2(),v,v)
    areas = np.sqrt(np.absolute(metric))
    v1_vecs = v1_vecs*(areas/v1_lengths).reshape(-1,1) @ Rz.T @ Rx.T
    v2_vecs = v2_vecs*(areas/v2_lengths).reshape(-1,1) @ Rz.T @ Rx.T

    end_points_1 = rotated_data + v1_vecs
    end_points_2 = rotated_data + v2_vecs

    all_tuples_x = [[a[0], a[1], a[2], None] for a in zip(end_points_1[:,0], rotated_data[:,0], end_points_2[:,0])]
    all_tuples_y = [[a[0], a[1], a[2], None] for a in zip(end_points_1[:,1], rotated_data[:,1], end_points_2[:,1])]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[p for l in all_tuples_x for p in l],
                            y=[p for l in all_tuples_y for p in l],
                            fill="toself",
                            fillcolor='gray',
                            marker=dict(size = 7, 
                                         color='black',
                                         colorscale=["gray", "black"])))
    fig.add_trace(go.Scatter(x = rotated_data[:,0], 
                             y = rotated_data[:,1], 
                             mode= 'markers', 
                             marker=dict(size = 7, 
                                         color=metric,
                                         colorscale=["gray", "black"],
                                         cmax = 0.1*metric.max(),
                                         cmin = metric.min())))
    fig.update_layout(width=800, 
                    height=500, 
                    showlegend=False, 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    xaxis_visible=False, 
                    yaxis_visible=False,
                    xaxis_range = [rotated_data[:,0].min() - 0.05, rotated_data[:,0].max() + 0.05],
                    yaxis_range = [rotated_data[:,1].min() - 0.05, rotated_data[:,1].max() + 0.05])
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    return fig