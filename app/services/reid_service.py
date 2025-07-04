import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from boxmot.appearance.reid.auto_backend import ReidAutoBackend
from app.core.config import settings

class ReIDService:
    def __init__(self, reid_model_path: str = None, device: int = None):
        self.device = device or settings.DEVICE
        self.reid_model_path = reid_model_path or settings.REID_MODEL_PATH
        self.reid_model = ReidAutoBackend(
            weights=Path(self.reid_model_path),
            device=self.device,
            half=True if self.device == 0 else False
        ).model

    def extract_features(self, image_path: str) -> Optional[np.ndarray]:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        h, w = img.shape[:2]
        bbox = np.array([[0, 0, w, h]])
        
        with torch.no_grad():
            features = self.reid_model.get_features(bbox, img)
            
        return features[0]

    def get_first_image_from_each_id(self, crops_dir: str) -> Dict[int, Path]:
        crops_path = Path(crops_dir)
        id_images = {}
        
        if not crops_path.exists():
            return id_images
        
        id_folders = [f for f in crops_path.iterdir() if f.is_dir() and f.name.startswith('id_')]
        
        for id_folder in sorted(id_folders):
            tracking_id = int(id_folder.name.split('_')[1])
            crop_path = id_folder / "crop.jpg"
            
            if crop_path.exists():
                id_images[tracking_id] = crop_path
        return id_images

    def extract_features_batch(self, id_images: Dict[int, Path]) -> Tuple[np.ndarray, List[int]]:
        features_list = []
        id_list = []
        
        for id_num, image_path in id_images.items():
            features = self.extract_features(image_path)
            if features is not None:
                features_list.append(features)
                id_list.append(id_num)
        
        if not features_list:
            return np.array([]), []
            
        features_array = np.vstack(features_list)
        return features_array, id_list

    def cluster_identities(self, features: np.ndarray, id_list: List[int], 
                         n_clusters: Optional[int] = None, 
                         distance_threshold: float = 0.5) -> Dict:
        if len(features) == 0:
            return {}
        
        similarity_matrix = cosine_similarity(features)
        distance_matrix = 1 - similarity_matrix
        
        if n_clusters is not None:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
        else:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric='precomputed',
                linkage='average'
            )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        clusters = {}
        for i, cluster_id in enumerate(cluster_labels):
            if cluster_id not in clusters:
                clusters[int(cluster_id)] = []
            clusters[int(cluster_id)].append(id_list[i])
        
        tracking_to_cluster = {}
        for i, cluster_id in enumerate(cluster_labels):
            tracking_id = id_list[i]
            tracking_to_cluster[tracking_id] = int(cluster_id)
        return {
            'cluster_labels': cluster_labels.tolist(),
            'id_list': id_list,
            'similarity_matrix': similarity_matrix.tolist(),
            'distance_matrix': distance_matrix.tolist(),
            'n_clusters_found': len(np.unique(cluster_labels)),
            'clusters': clusters,
            'tracking_to_cluster': tracking_to_cluster
        }

    def process_clustering(self, crops_dir: str, distance_threshold: float = 0.2,
                          n_clusters: Optional[int] = None) -> Optional[Dict]:
        id_images = self.get_first_image_from_each_id(crops_dir)
        if not id_images:
            return None
        
        features, id_list = self.extract_features_batch(id_images)
        if len(features) == 0:
            return None
        
        return self.cluster_identities(features, id_list, n_clusters, distance_threshold)