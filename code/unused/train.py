import os
import argparse
import logging
import yaml
from ultralytics import YOLO
import shutil
import sys
import requests

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

def get_model_path(model_name):
    """Get path to model, download if doesn't exist"""
    # Get absolute path to models directory
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, model_name)
    if not os.path.exists(model_path):
        logging.info(f"Downloading {model_name} to models directory...")
        # Download model using YOLO's download method
        model = YOLO(model_name)
        
        # Get the downloaded model path (usually in current directory)
        downloaded_path = os.path.join(os.getcwd(), model_name)
        
        # Move the model to our models directory
        if os.path.exists(downloaded_path):
            os.rename(downloaded_path, model_path)
            logging.info(f"Model moved to {model_path}")
        else:
            logging.error(f"Downloaded model not found at {downloaded_path}")
            return model_name  # Fallback to default location
    else:
        logging.info(f"Loading existing model from {model_path}")
    
    return model_path

def validate_yaml(yaml_path):
    """Validate the YAML file exists and has required fields"""
    # Get absolute path to data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Handle remote URLs
    if yaml_path.startswith(('http://', 'https://')):
        try:
            import requests
            yaml_name = yaml_path.split('/')[-1]
            local_yaml = os.path.join(data_dir, yaml_name)
            
            if not os.path.exists(local_yaml):
                logging.info(f"Downloading YAML from {yaml_path}")
                response = requests.get(yaml_path)
                response.raise_for_status()
                
                with open(local_yaml, 'w') as f:
                    f.write(response.text)
                logging.info(f"YAML downloaded to {local_yaml}")
            
            yaml_path = local_yaml
        except Exception as e:
            logging.error(f"Error downloading YAML: {str(e)}")
            return yaml_path

    # If it's a built-in dataset, copy its YAML to our data directory
    if yaml_path in ['coco.yaml', 'voc.yaml', 'mnist.yaml']:
        # First try to download using YOLO
        try:
            model = YOLO('yolov8n.pt')  # temporary model to access dataset
            dataset_yaml = os.path.join(data_dir, yaml_path)
            if not os.path.exists(dataset_yaml):
                # The dataset YAML will be downloaded to current directory
                temp_yaml = yaml_path
                if os.path.exists(temp_yaml):
                    shutil.move(temp_yaml, dataset_yaml)
                    logging.info(f"Dataset config moved to {dataset_yaml}")
            return dataset_yaml
        except Exception as e:
            logging.error(f"Error downloading dataset config: {str(e)}")
            return yaml_path
    
    # For custom YAML files, validate and copy to data directory
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Data config file not found: {yaml_path}")
    
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            
        required_fields = ['path', 'train', 'val', 'names']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            raise ValueError(f"YAML file missing required fields: {missing_fields}")
        
        # Copy custom YAML to data directory if it's not already there
        yaml_name = os.path.basename(yaml_path)
        data_yaml_path = os.path.join(data_dir, yaml_name)
        if yaml_path != data_yaml_path:
            shutil.copy2(yaml_path, data_yaml_path)
            logging.info(f"Copied dataset config to {data_yaml_path}")
            
        return data_yaml_path
            
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {str(e)}")

def download_dataset(yaml_path):
    """Download dataset if needed based on YAML configuration"""
    try:
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
            
        if 'download' in yaml_data:
            logging.info("Dataset download script found in YAML")
            
            # Create a temporary Python file with the download script
            script_path = os.path.join(os.path.dirname(yaml_path), '_download_temp.py')
            
            # Prepare the script with the necessary context
            script_content = f"""
import yaml

# Load the YAML data
yaml_data = {yaml_data}

# Make yaml variable available for the download script
yaml = yaml_data

{yaml_data['download']}
"""
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Execute the download script
            logging.info("Downloading dataset... This may take a while...")
            import subprocess
            subprocess.run([sys.executable, script_path], check=True)
            
            # Clean up
            os.remove(script_path)
            logging.info("Dataset download completed")
            
    except Exception as e:
        logging.error(f"Error downloading dataset: {str(e)}")
        raise

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO model on custom dataset')
    
    # Required arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data configuration file (e.g., "coco.yaml")')
    parser.add_argument('--base_model', type=str, required=True,
                       help='Base YOLO model to start training from (e.g., "yolov8n.pt")')
    parser.add_argument('--save_name', type=str, required=True,
                       help='Name to save the trained model as (e.g., "my_model.pt")')
    
    # Optional training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for training (default: 640)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--device', type=str, default='',
                       help='Device to train on (e.g., "0" or "cpu") (default: auto)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of worker threads (default: 8)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get absolute path to models directory
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Validate and get path to YAML file
    try:
        data_yaml_path = validate_yaml(args.data)
        logging.info(f"Data config validated: {data_yaml_path}")
        
        # Download dataset if needed
        download_dataset(data_yaml_path)
    except Exception as e:
        logging.error(f"Error in data config: {str(e)}")
        return
    
    # Get base model path
    base_model_path = get_model_path(args.base_model)
    
    # Load base model
    try:
        model = YOLO(base_model_path)
        logging.info(f"Loaded base model: {base_model_path}")
    except Exception as e:
        logging.error(f"Error loading base model: {str(e)}")
        return
    
    # Prepare save path
    save_path = os.path.join(models_dir, args.save_name)
    if not args.save_name.endswith('.pt'):
        save_path += '.pt'
    
    # Check if save name already exists
    if os.path.exists(save_path):
        logging.warning(f"Model {save_path} already exists. Please choose a different name.")
        return
    
    # Configure training parameters
    train_args = {
        'data': data_yaml_path,  # Use the validated YAML path
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'workers': args.workers,
        'device': args.device,
        'name': os.path.splitext(args.save_name)[0],  # Remove .pt for run name
        'project': models_dir,  # Save in models directory
        'exist_ok': True  # Overwrite existing files
    }
    
    # Start training
    logging.info("Starting training with parameters:")
    for k, v in train_args.items():
        logging.info(f"  {k}: {v}")
    
    try:
        results = model.train(**train_args)
        
        # Move the best model to the models directory with the desired name
        best_model = os.path.join(models_dir, args.save_name.split('.')[0], 'weights', 'best.pt')
        if os.path.exists(best_model):
            os.rename(best_model, save_path)
            logging.info(f"Best model saved as: {save_path}")
        else:
            logging.warning("Best model not found. Check training outputs.")
            
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        return
    
    logging.info("Training completed successfully!")

if __name__ == "__main__":
    main() 