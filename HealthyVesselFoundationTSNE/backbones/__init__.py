"""Backbone adapters for HealthyVesselFoundationTSNE.

Adapters expose:

    embed_point_clouds(points: np.ndarray, config: Mapping[str, Any], device: str) -> np.ndarray

The main pipeline passes point clouds as [B, N, 3] float32 arrays.
"""

