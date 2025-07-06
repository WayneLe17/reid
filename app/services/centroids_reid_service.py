import numpy as np
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from torchvision import transforms
from PIL import Image
from app.core.config import settings
# Import from centroids-reid
from app.libs.centroids_reid.config.defaults import _C as cfg
from app.libs.centroids_reid.train_ctl_model import CTLModel
from app.libs.centroids_reid.inference.inference_utils import (
    ImageDataset,
    ImageFolderWithPaths,
    calculate_centroids as ctl_calculate_centroids,
    create_pid_path_index,
    make_inference_data_loader,
    run_inference,
    _inference,
    pil_loader
)
from app.libs.centroids_reid.datasets.transforms import ReidTransforms


class CentroidsReIDService:
    
    def __init__(
        self, 
        model_path: str = None,
        device: Union[str, int] = None,
        use_centroids: bool = True,
        normalize_features: bool = True,
        distance_func: str = 'cosine',
        config_file: str = None
    ):
        self.device = device or settings.DEVICE
        self.model_path = model_path or settings.REID_MODEL_PATH
        self.use_centroids = use_centroids
        self.normalize_features = normalize_features
        self.distance_func = distance_func
        
        # Configure CTL model
        self.cfg = self._configure_model(config_file)
        
        # Load model
        self.model = self._load_ctl_model()
        
        # Set up transforms using ReidTransforms from centroids-reid
        self.transforms_base = ReidTransforms(self.cfg)
        self.transform = self.transforms_base.build_transforms(is_train=False)
        
        # Function to extract PID from path (for centroid calculation)
        self.extract_pid_func = lambda x: Path(x).parent.name
    
    def _configure_model(self, config_file: str = None) -> cfg:
        """Configure the CTL model using YACS config."""
        # Reset to defaults
        cfg.defrost()
        
        # Load config file if provided
        if config_file and Path(config_file).exists():
            cfg.merge_from_file(config_file)
        
        # Override with our settings
        cfg.MODEL.PRETRAIN_PATH = self.model_path
        cfg.MODEL.USE_CENTROIDS = self.use_centroids
        cfg.TEST.FEAT_NORM = self.normalize_features
        cfg.SOLVER.DISTANCE_FUNC = self.distance_func
        cfg.INPUT.SIZE_TEST = [256, 128]
        cfg.TEST.IMS_PER_BATCH = 32
        cfg.DATALOADER.NUM_WORKERS = 4
        
        # Set device
        if isinstance(self.device, int):
            cfg.GPU_IDS = [self.device]
        else:
            cfg.GPU_IDS = []
        
        cfg.freeze()
        return cfg
    
    def _load_ctl_model(self) -> CTLModel:
        """Load the CTL model from checkpoint."""
        model_path = Path(self.model_path)
        
        if model_path.exists():
            print(f"Loading CTL model from: {model_path}")
            try:
                # Load model using CTLModel.load_from_checkpoint
                model = CTLModel.load_from_checkpoint(
                    self.cfg.MODEL.PRETRAIN_PATH,
                    cfg=self.cfg
                )
                
                # Set to eval mode and move to device
                use_cuda = torch.cuda.is_available() and self.cfg.GPU_IDS
                if use_cuda:
                    model = model.cuda()
                model.eval()
                
                return model
            except Exception as e:
                print(f"Error loading CTL model: {e}")
                print("Falling back to creating model from scratch")
        
        # Create model from scratch
        print("Creating new CTL model")
        model = CTLModel(cfg=self.cfg)
        
        use_cuda = torch.cuda.is_available() and self.cfg.GPU_IDS
        if use_cuda:
            model = model.cuda()
        model.eval()
        
        return model
    
    def _extract_single_image_features(self, image_path: str) -> Optional[np.ndarray]:
        """Extract features from a single image using CTL model inference."""
        try:
            # Load image
            image = pil_loader(str(image_path))
            
            # Apply transforms
            if self.transform is not None:
                image_tensor = self.transform(image)
            else:
                # Fallback to basic transforms
                image_tensor = transforms.ToTensor()(image)
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            # Prepare batch (image, label, path)
            batch = (image_tensor, torch.tensor([0]), [str(image_path)])
            
            # Run inference
            use_cuda = torch.cuda.is_available() and self.cfg.GPU_IDS
            features, _ = _inference(self.model, batch, use_cuda)
            
            # Convert to numpy
            features = features[0].detach().cpu().numpy()
            
            # Normalize if needed
            if self.normalize_features:
                norm = np.linalg.norm(features)
                if norm > 0:
                    features = features / norm
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def extract_features(self, image_path: str) -> Optional[np.ndarray]:
        """Extract features from an image path."""
        return self._extract_single_image_features(image_path)
    
    def extract_features_from_directory(self, directory: str, use_subfolders: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from all images in a directory using CTL inference pipeline."""
        # Choose dataset type
        if use_subfolders:
            dataset_type = ImageFolderWithPaths
        else:
            dataset_type = ImageDataset
        
        # Create data loader
        val_loader = make_inference_data_loader(self.cfg, directory, dataset_type)
        
        if len(val_loader) == 0:
            return np.array([]), np.array([])
        
        # Run inference
        use_cuda = torch.cuda.is_available() and self.cfg.GPU_IDS
        embeddings, paths = run_inference(
            self.model, val_loader, self.cfg, print_freq=10, use_cuda=use_cuda
        )
        
        return embeddings, paths
    
    def get_all_images_per_id(self, crops_dir: str) -> Dict[int, List[Path]]:
        crops_path = Path(crops_dir)
        id_images = {}
        
        if not crops_path.exists():
            return id_images
        
        id_folders = [f for f in crops_path.iterdir() if f.is_dir() and f.name.startswith('id_')]
        
        for id_folder in sorted(id_folders):
            tracking_id = int(id_folder.name.split('_')[1])
            
            image_files = list(id_folder.glob("*.jpg")) + list(id_folder.glob("*.png"))
            
            if image_files:
                id_images[tracking_id] = image_files
                
        return id_images
    
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
    
    def compute_centroid(self, features_list: List[np.ndarray]) -> np.ndarray:
        if not features_list:
            return np.array([])
        
        features_array = np.vstack(features_list)
        centroid = np.mean(features_array, axis=0)
        
        if self.normalize_features:
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
                
        return centroid
    
    def compute_centroids_batch(self, embeddings: np.ndarray, paths: np.ndarray, 
                               extract_func: Callable = None) -> Tuple[np.ndarray, np.ndarray]:
        if extract_func is None:
            extract_func = self.extract_pid_func
        
        pid_path_index = create_pid_path_index(paths.tolist(), extract_func)
        
        centroids, pids = ctl_calculate_centroids(embeddings, pid_path_index)
        
        return centroids, pids
    
    def extract_features_batch(self, id_images: Dict[int, Union[Path, List[Path]]]) -> Tuple[np.ndarray, List[int]]:
        features_list = []
        id_list = []
        
        for id_num, image_paths in id_images.items():
            if isinstance(image_paths, Path):
                features = self.extract_features(image_paths)
                if features is not None:
                    features_list.append(features)
                    id_list.append(id_num)
            else:
                if self.use_centroids and len(image_paths) > 1:
                    id_features = []
                    for img_path in image_paths:
                        feat = self.extract_features(img_path)
                        if feat is not None:
                            id_features.append(feat)
                    
                    if id_features:
                        centroid = self.compute_centroid(id_features)
                        features_list.append(centroid)
                        id_list.append(id_num)
                else:
                    features = self.extract_features(image_paths[0])
                    if features is not None:
                        features_list.append(features)
                        id_list.append(id_num)
        
        if not features_list:
            return np.array([]), []
        
        features_array = np.vstack(features_list)
        return features_array, id_list
    
    def compute_distance_matrix(self, features: np.ndarray) -> np.ndarray:
        if self.distance_func == 'cosine':
            similarity_matrix = cosine_similarity(features)
            distance_matrix = 1 - similarity_matrix
        elif self.distance_func == 'euclidean':
            distance_matrix = euclidean_distances(features)
            if distance_matrix.max() > 0:
                distance_matrix = distance_matrix / distance_matrix.max()
        else:
            raise ValueError(f"Unknown distance function: {self.distance_func}")
        
        return distance_matrix
    
    def cluster_identities(
        self,
        features: np.ndarray,
        id_list: List[int],
        n_clusters: Optional[int] = None,
        distance_threshold: float = 0.5
    ) -> Dict:
        if len(features) == 0:
            return {}
        
        distance_matrix = self.compute_distance_matrix(features)
        
        if self.distance_func == 'cosine':
            similarity_matrix = 1 - distance_matrix
        else:
            similarity_matrix = 1 - distance_matrix
        
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
            'tracking_to_cluster': tracking_to_cluster,
            'distance_func': self.distance_func,
            'used_centroids': self.use_centroids
        }
    
    def process_clustering(
        self,
        crops_dir: str,
        distance_threshold: float = 0.2,
        n_clusters: Optional[int] = None
    ) -> Optional[Dict]:
        """Process clustering using CTL model inference pipeline."""
        crops_path = Path(crops_dir)
        if not crops_path.exists():
            return None
        
        # Extract features from all images in subdirectories
        embeddings, paths = self.extract_features_from_directory(str(crops_path), use_subfolders=True)
        
        if len(embeddings) == 0:
            return None
        
        # Extract tracking IDs from paths (id_X folders)
        id_list = []
        features_list = []
        
        if self.use_centroids:
            # Group features by tracking ID and compute centroids
            id_features_map = {}
            
            for embedding, path in zip(embeddings, paths):
                # Extract tracking ID from path like: .../id_1/crop.jpg
                path_obj = Path(path)
                if path_obj.parent.name.startswith('id_'):
                    tracking_id = int(path_obj.parent.name.split('_')[1])
                    
                    if tracking_id not in id_features_map:
                        id_features_map[tracking_id] = []
                    id_features_map[tracking_id].append(embedding)
            
            # Compute centroids for each ID
            for tracking_id in sorted(id_features_map.keys()):
                centroid = self.compute_centroid(id_features_map[tracking_id])
                features_list.append(centroid)
                id_list.append(tracking_id)
            
            features = np.vstack(features_list)
        else:
            # Use first image per ID
            seen_ids = set()
            
            for embedding, path in zip(embeddings, paths):
                path_obj = Path(path)
                if path_obj.parent.name.startswith('id_'):
                    tracking_id = int(path_obj.parent.name.split('_')[1])
                    
                    if tracking_id not in seen_ids:
                        seen_ids.add(tracking_id)
                        features_list.append(embedding)
                        id_list.append(tracking_id)
            
            if not features_list:
                return None
                
            features = np.vstack(features_list)
        
        return self.cluster_identities(features, id_list, n_clusters, distance_threshold)