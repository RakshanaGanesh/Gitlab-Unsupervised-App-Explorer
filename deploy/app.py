from flask import Flask, request, jsonify, render_template
import pickle
import networkx as nx
import numpy as np
import plotly
import plotly.graph_objects as go
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
import json
import plotly.colors

app = Flask(__name__)

# Load trained models and graphs
models = {
    "Category + Installs": "category_installs.pkl",
    "Category + Rating": "category_rating.pkl",
    "Category + Size": "category_size.pkl",
    "Category + Reviews": "category_reviews.pkl",
    "Combined": "Combined.pkl"
}

graphs = {
    "Category + Installs": "category_installs.graphml",
    "Category + Rating": "category_rating.graphml",
    "Category + Size": "category_size.graphml",
    "Category + Reviews": "category_reviews.graphml",
    "Combined": "Combined.graphml"
}

model_objects = {name: pickle.load(open(models[name], "rb")) for name in models}
graph_objects = {name: nx.read_graphml(graphs[name]) for name in graphs}

# Function to find similar apps
def find_similar_apps(input_app, model, G, top_n=9):
    input_node = None
    for node, data in G.nodes(data=True):
        if data.get("name", "").lower() == input_app.lower():
            input_node = str(node)
            break

    if input_node is None:
        return "App not found.", None

    input_embedding = model.wv[input_node]

    similarities = []
    for node in G.nodes():
        if str(node) != input_node:
            sim = 1 - cosine(input_embedding, model.wv[str(node)])
            similarities.append((G.nodes[node]["name"], sim, str(node)))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n], input_node

# Function to generate similarity graph

def plot_similarity_graph(input_app, selected_model):
    G = graph_objects[selected_model]
    model = model_objects[selected_model]

    similar_apps, selected_node = find_similar_apps(input_app, model, G, top_n=9)

    if similar_apps == "App not found.":
        return None

    selected_nodes = [selected_node] + [app[2] for app in similar_apps]
    sub_G = G.subgraph(selected_nodes)
    pos = nx.kamada_kawai_layout(sub_G)  # Better spacing

    # Attributes based on selected model
    model_attributes = {
        "Category + Installs": ["Category", "Installs"],
        "Category + Rating": ["Category", "Rating"],
        "Category + Size": ["Category", "Size"],
        "Category + Reviews": ["Category", "Reviews"],
        "Combined": ["Category", "Installs", "Rating", "Size", "Reviews"]
    }
    attributes_to_show = model_attributes[selected_model]

    node_x, node_y, node_labels, node_sizes, node_colors, node_hovertext = [], [], [], [], [], []
    edge_x, edge_y, edge_widths, edge_hovertext = [], [], [], []
    edge_annotations = []

    max_similarity = max(similar_apps, key=lambda x: x[1])[1] if similar_apps else 1
    color_scale = "Plasma"  # More vibrant colors

    for node, coord in pos.items():
        node_x.append(coord[0])
        node_y.append(coord[1])
        node_labels.append(G.nodes[node]["name"])
        node_sizes.append(35 if node == selected_node else 16)
        
        similarity_score = next((sim[1] for sim in similar_apps if sim[2] == node), 0)
        color_intensity = similarity_score / max_similarity
        node_colors.append(plotly.colors.sample_colorscale(color_scale, [color_intensity])[0])

        hover_details = {attr: G.nodes[node].get(attr, "N/A") for attr in attributes_to_show}
        hover_text = "<br>".join([f"<b>{key}:</b> {value}" for key, value in hover_details.items()])
        node_hovertext.append(hover_text)

    for sim_app in similar_apps:
        app_node = sim_app[2]
        similarity_score = sim_app[1]

        if sub_G.has_edge(selected_node, app_node):
            x0, y0 = pos[selected_node]
            x1, y1 = pos[app_node]

            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_widths.append(similarity_score * 6)
            edge_hovertext.append(f"<b>Similarity:</b> {similarity_score:.2f}")

            edge_annotations.append(
                dict(
                    x=(x0 + x1) / 2 + 0.02,  # Slight shift
                    y=(y0 + y1) / 2 + 0.02,
                    text=f"{similarity_score:.2f}",
                    showarrow=False,
                    font=dict(size=14, color="black", family="Arial Black"),  # Black text
                    xanchor="center",
                    yanchor="middle"
                )
            )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=2.5, color="darkgray"), mode="lines", hoverinfo="text", text=edge_hovertext))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", text=node_labels, marker=dict(size=node_sizes, color=node_colors, line=dict(color="black", width=1.5)), textposition="top center", hoverinfo="text", hovertext=node_hovertext))

    fig.update_layout(
        title=f"🔍 Similarity Graph for <b>{input_app}</b> using <b>{selected_model}</b>",
        annotations=edge_annotations,
        showlegend=False,
        plot_bgcolor="lightgray",  # Improved background
        paper_bgcolor="lightgray",
        font=dict(color="black"),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)




@app.route("/")
def home():
    return render_template("index.html")

@app.route("/find_similar", methods=["POST"])
def find_similar():
    data = request.json
    app_name = data.get("app_name")
    selected_model = data.get("model")

    if not app_name or selected_model not in model_objects:
        return jsonify({"error": "Invalid input."})

    graph_json = plot_similarity_graph(app_name, selected_model)
    
    if not graph_json:
        return jsonify({"error": "App not found."})

    similar_apps, _ = find_similar_apps(app_name, model_objects[selected_model], graph_objects[selected_model])
    
    return jsonify({"similar_apps": similar_apps, "graph": graph_json})

if __name__ == "__main__":
    app.run(debug=True)
