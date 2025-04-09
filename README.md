# Unsupervised-App-Explorer
# App Similarity Finder using Unsupervised Learning

This project uses **unsupervised learning** techniques to find and visualize similar Android apps from the Google Play Store. By combining graph embeddings (Node2Vec), clustering (Spectral Clustering), and interactive visualization (Plotly), the system recommends similar apps based on various feature combinations.

## Features

- Find similar apps using:
  - Category + Rating
  - Category + Size
  - Category + Reviews
  - Combined Features (weighted similarity)
- Graph-based unsupervised learning using Node2Vec
- Spectral Clustering for enhanced visualization and exploration
- Interactive web interface built with Flask + Plotly
- Hover tooltips showing detailed app info and similarity scores

## Techniques Used

- **Node2Vec**: To generate graph embeddings for apps
- **Cosine Similarity**: To compute similarity between app vectors
- **Spectral Clustering**: To assign cluster colors for nodes in the graph
- **Plotly**: For dynamic graph visualization
- **Flask**: To build the web server for user interaction
