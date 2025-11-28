# @title ArchNet Real Dataset Benchmarking Pipeline
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import json
from pathlib import Path
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print("✅ Libraries imported successfully")

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration parameters"""
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IMG_CHANNELS = 3
    BATCH_SIZE = 32
    EPOCHS = 5
    SAMPLE_SIZE = 50 # Images per class for quick testing (FIXED U+00A0 ERROR)

    # Dataset configurations
    DATASETS = {
        'chest_xray': {
            'kaggle_ref': 'paultimothymooney/chest-xray-pneumonia',
            'classes': ['NORMAL', 'PNEUMONIA'],
            'task_type': 'binary'
        },
        'covid': {
            'kaggle_ref': 'tawsifurrahman/covid19-radiography-database',
            'classes': ['COVID', 'Normal', 'Viral Pneumonia'],
            'task_type': 'multiclass'
        }
    }

config = Config()

# =============================================================================
# KAGGLE SETUP (Colab Optimized)
# =============================================================================

def setup_kaggle():
    """Setup Kaggle API in Colab"""
    print("🔧 Setting up Kaggle API...")

    # This will prompt for Kaggle JSON upload in Colab
    if not os.path.exists('/root/.kaggle/kaggle.json'):
        print("📁 Please upload your kaggle.json file when prompted")
        from google.colab import files
        uploaded = files.upload()

        # Move to proper location
        os.makedirs('/root/.kaggle', exist_ok=True)
        # Using standard shell commands via ! for Colab environment
        !mv kaggle.json /root/.kaggle/
        !chmod 600 /root/.kaggle/kaggle.json

    print("✅ Kaggle API setup complete")

# =============================================================================
# DATASET DOWNLOADER
# =============================================================================

class RealDatasetLoader:
    """Downloads and processes real datasets from Kaggle"""

    def __init__(self, config):
        self.config = config
        self.dataset_info = {}

    def download_dataset(self, dataset_name):
        """Download specific dataset from Kaggle"""
        if dataset_name not in self.config.DATASETS:
            print(f"❌ Dataset {dataset_name} not configured")
            return False

        dataset_config = self.config.DATASETS[dataset_name]
        print(f"📥 Downloading {dataset_name} dataset...")

        try:
            # Download dataset using standard shell command
            !kaggle datasets download -d {dataset_config['kaggle_ref']} -p /content/data/ --unzip
            print(f"✅ {dataset_name} downloaded successfully")
            return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False

    def discover_dataset_structure(self, dataset_name):
        """Discover the actual structure of downloaded dataset"""
        base_path = '/content/data'

        print(f"🔍 Discovering {dataset_name} structure...")

        structure = {}
        for root, dirs, files in os.walk(base_path):
            # Skip macOS metadata directories
            if '__MACOSX' in root:
                continue

            # Look for image directories
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                rel_path = os.path.relpath(root, base_path)
                structure[rel_path] = {
                    'path': root,
                    'file_count': len(image_files),
                    'sample_files': image_files[:3] # First 3 files
                }

        # Print discovered structure
        print(f"📁 Discovered structure for {dataset_name}:")
        for path, info in structure.items():
            print(f"   📂 {path}: {info['file_count']} images")

        return structure

    def load_images_from_structure(self, dataset_name, structure):
        """Load actual images from discovered structure"""
        print(f"🖼️ Loading images for {dataset_name}...")

        image_paths = []
        labels = []
        class_mapping = {}

        dataset_config = self.config.DATASETS[dataset_name]
        expected_classes = dataset_config['classes']

        # Map class names from directory structure
        for path, info in structure.items():
            for class_name in expected_classes:
                # Use 'in' to find the class name within the directory path
                if class_name.lower() in path.lower():
                    if class_name not in class_mapping.values():
                        class_idx = len(class_mapping)
                        class_mapping[class_idx] = class_name

                    # Get actual image files
                    image_files = [f for f in os.listdir(info['path'])
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                    # Take sample of images
                    sample_size = min(self.config.SAMPLE_SIZE, len(image_files))
                    sampled_files = image_files[:sample_size]

                    for img_file in sampled_files:
                        image_paths.append(os.path.join(info['path'], img_file))
                        labels.append([k for k, v in class_mapping.items() if v == class_name][0])

        print(f"📊 Loaded {len(image_paths)} images for {len(class_mapping)} classes")
        return image_paths, labels, class_mapping

# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def preprocess_image(image_path, label):
    """Load and preprocess actual image"""
    try:
        # Read and decode image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)

        # Convert to float and resize
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [config.IMG_HEIGHT, config.IMG_WIDTH])

        # Ensure proper shape
        image.set_shape([config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS])

        return image, label

    except Exception as e:
        print(f"⚠️ Error processing {image_path}: {e}")
        # Return blank image as fallback
        blank_image = tf.zeros([config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS], dtype=tf.float32)
        return blank_image, label

def create_tf_dataset(image_paths, labels, augment=False):
    """Create TensorFlow dataset from real images"""
    if len(image_paths) == 0:
        return None

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        augmentation = keras.Sequential([
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomFlip("horizontal"),
        ])
        dataset = dataset.map(lambda x, y: (augmentation(x, training=True), y),
                              num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

def create_cnn_model(architecture_name, num_classes):
    """Create CNN model with transfer learning"""
    print(f"🔨 Building {architecture_name}...")

    if architecture_name == 'VGG16':
        base_model = applications.VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)
        )
    elif architecture_name == 'ResNet50':
        base_model = applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)
        )
    elif architecture_name == 'DenseNet121':
        base_model = applications.DenseNet121(
            include_top=False,
            weights='imagenet',
            input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)
        )
    elif architecture_name == 'EfficientNetB0':
        base_model = applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture_name}")

    base_model.trainable = False

    # Build classifier
    inputs = keras.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)

    # Output layer
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=loss,
        metrics=metrics
    )

    print(f"✅ {architecture_name} built - {model.count_params():,} parameters")
    return model

# =============================================================================
# BENCHMARKING SYSTEM
# =============================================================================

class BenchmarkManager:
    """Manages model training and evaluation"""

    def __init__(self, config):
        self.config = config
        self.results = []

    def calculate_metrics(self, model, test_data, num_classes):
        """Calculates Accuracy, Precision, Recall, F1-score from test data"""
        if test_data is None:
            return 0.0, 0.0, 0.0, 0.0 # Return zeros if no test data

        y_true = []
        y_pred = []

        print("🔍 Running model predictions on test dataset...")

        # Loop through test dataset
        for images, labels in test_data:
            preds = model.predict(images, verbose=0)
            labels_np = labels.numpy()

            # Binary classification (num_classes == 2 is the condition for binary in problem setup)
            if num_classes == 2:
                # Sigmoid output (single value), threshold at 0.5
                preds_np = (preds > 0.5).astype(int).flatten()
                labels_np = labels_np.astype(int).flatten()
            # Multiclass classification
            else:
                # Softmax output, take the index of the highest probability
                preds_np = np.argmax(preds, axis=1)

            y_true.extend(labels_np)
            y_pred.extend(preds_np)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if len(y_true) == 0:
            return 0.0, 0.0, 0.0, 0.0

        # Compute metrics
        # Use 'weighted' average for multiclass/binary metrics to account for class imbalance
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        return accuracy, precision, recall, f1


    def train_and_evaluate(self, model, model_name, dataset_name, train_data, val_data, test_data, num_classes):
        """Train and evaluate a model"""
        print(f"🎯 Training {model_name} on {dataset_name}...")

        if train_data is None:
            print(f"❌ No training data for {dataset_name}")
            return None, None

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=3, restore_best_weights=True
            )
        ]

        # Train model
        start_time = time.time()
        try:
            history = model.fit(
                train_data,
                validation_data=val_data,
                epochs=self.config.EPOCHS,
                callbacks=callbacks,
                verbose=1
            )
            training_time = time.time() - start_time
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None, None

        # Calculate true evaluation metrics
        test_accuracy, precision, recall, f1 = self.calculate_metrics(model, test_data, num_classes)

        # Calculate loss (This uses the compiled model's loss metric)
        if test_data is not None:
            try:
                # Keras evaluate returns (loss, metrics...)
                eval_results = model.evaluate(test_data, verbose=0)
                # Loss is always the first element
                test_loss = eval_results[0]
            except Exception as e:
                print(f"⚠️ Error evaluating loss: {e}")
                test_loss = 0.0
        else:
            test_loss = 0.0

        # Model size and inference time
        model_size = model.count_params()
        inference_time = self.measure_inference_time(model, test_data)

        # Store results
        result = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': training_time,
            'inference_time': inference_time,
            'model_parameters': model_size,
            'model_size_mb': (model_size * 4) / (1024 * 1024) # Assuming float32 parameters
        }

        self.results.append(result)

        print(f"✅ {model_name} on {dataset_name}:")
        print(f"   📊 Accuracy: {test_accuracy:.4f}, F1: {f1:.4f}")
        print(f"   ⚡ Training: {training_time:.1f}s, Inference: {inference_time:.4f}s")

        return result, history

    def measure_inference_time(self, model, test_data, num_samples=50):
        """Measure inference time"""
        if test_data is None:
            return 0.01 # Default small value if no test data

        try:
            times = []
            # Take one batch for inference time measurement
            for images, _ in test_data.take(1):
                start_time = time.time()
                _ = model.predict(images, verbose=0) # Make predictions
                batch_time = (time.time() - start_time) / len(images) # Time per image in batch
                times.append(batch_time)

            return np.mean(times) if times else 0.01
        except Exception as e:
            print(f"⚠️ Error measuring inference time: {e}")
            return 0.01 # Fallback

# =============================================================================
# VISUALIZATION SYSTEM
# =============================================================================

class VisualizationManager:
    """Handles all visualizations"""

    def __init__(self):
        self.figures = [] # To keep track of figures if needed

    def plot_sample_images(self, dataset, dataset_name, class_mapping):
        """Plot sample images from dataset"""
        print(f"📊 Plotting sample images for {dataset_name}...")
        

        plt.figure(figsize=(15, 5))

        # Take one batch
        for images, labels in dataset.take(1):
            for i in range(min(8, len(images))):
                plt.subplot(2, 4, i+1)
                image = images[i].numpy()
                label = labels[i].numpy()

                # Clip and display (images are float32 in [0,1])
                image = np.clip(image, 0, 1)
                plt.imshow(image)
                # Handle class mapping for display
                class_name = class_mapping.get(label, f"Class {label}")
                plt.title(f'{class_name}')
                plt.axis('off')

        plt.suptitle(f'{dataset_name} - Sample Images', fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_training_history(self, history, model_name, dataset_name):
        """Plot training history"""
        if history is None:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        

        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title(f'{model_name} - Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Loss plot
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title(f'{model_name} - Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_benchmark_results(self, results_df):
        """Plot benchmark comparison"""
        if results_df.empty:
            print("No results to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Benchmarking', fontsize=18, y=1.02)
        

        # Accuracy comparison
        sns.barplot(x='model_name', y='test_accuracy', hue='dataset_name', data=results_df, ax=axes[0,0], palette='viridis')
        axes[0,0].set_title('Model Accuracy Comparison')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].legend(title='Dataset', loc='lower right')

        # Inference time comparison
        sns.barplot(x='model_name', y='inference_time', hue='dataset_name', data=results_df, ax=axes[0,1], palette='plasma')
        axes[0,1].set_title('Inference Time Comparison')
        axes[0,1].set_ylabel('Time (seconds)')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].legend(title='Dataset', loc='upper right')

        # Model size comparison
        sns.barplot(x='model_name', y='model_size_mb', hue='dataset_name', data=results_df, ax=axes[1,0], palette='magma')
        axes[1,0].set_title('Model Size Comparison')
        axes[1,0].set_ylabel('Size (MB)')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].legend(title='Dataset', loc='upper right')

        # F1-score comparison
        sns.barplot(x='model_name', y='f1_score', hue='dataset_name', data=results_df, ax=axes[1,1], palette='cividis')
        axes[1,1].set_title('F1-Score Comparison')
        axes[1,1].set_ylabel('F1 Score')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].legend(title='Dataset', loc='lower right')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent suptitle overlap
        plt.show()

# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================

def main():
    """Main execution pipeline"""
    print("🚀 Starting ArchNet Real Dataset Pipeline...")

    # Setup
    setup_kaggle()

    # Initialize components
    data_loader = RealDatasetLoader(config)
    benchmark_manager = BenchmarkManager(config)
    visualizer = VisualizationManager()

    all_results = []
    training_histories = {}

    # Process each dataset
    for dataset_name in ['chest_xray', 'covid']:
        print(f"\n{'='*60}")
        print(f"📊 PROCESSING {dataset_name.upper()}")
        print(f"{'='*60}")

        # Download dataset
        if not data_loader.download_dataset(dataset_name):
            print(f"❌ Skipping {dataset_name} due to download failure")
            continue

        # Discover structure and load images
        structure = data_loader.discover_dataset_structure(dataset_name)
        image_paths, labels, class_mapping = data_loader.load_images_from_structure(
            dataset_name, structure
        )

        if len(image_paths) == 0:
            print(f"❌ No images found for {dataset_name}")
            continue

        # Split data
        try:
            # Use stratification if enough samples exist for all classes
            train_paths, test_paths, train_labels, test_labels = train_test_split(
                image_paths, labels, test_size=0.3, random_state=42, stratify=labels
            )
            val_paths, test_paths, val_labels, test_labels = train_test_split(
                test_paths, test_labels, test_size=0.5, random_state=42, stratify=test_labels
            )
        except ValueError as e:
            # Fallback to non-stratified split if a class is too small
            print(f"⚠️ Not enough samples for stratification in {dataset_name}: {e}. Skipping stratification.")
            train_paths, test_paths, train_labels, test_labels = train_test_split(
                image_paths, labels, test_size=0.3, random_state=42
            )
            val_paths, test_paths, val_labels, test_labels = train_test_split(
                test_paths, test_labels, test_size=0.5, random_state=42
            )

        num_classes = len(class_mapping)

        # Create datasets
        train_dataset = create_tf_dataset(train_paths, train_labels, augment=True)
        val_dataset = create_tf_dataset(val_paths, val_labels, augment=False)
        test_dataset = create_tf_dataset(test_paths, test_labels, augment=False)

        # Visualize samples (only for training data)
        if train_dataset:
            visualizer.plot_sample_images(train_dataset, dataset_name, class_mapping)

        # Benchmark models
        architectures = ['VGG16', 'ResNet50', 'DenseNet121', 'EfficientNetB0']

        dataset_histories = {}

        for arch in architectures:
            try:
                # Create and train model
                model = create_cnn_model(arch, num_classes)
                result, history = benchmark_manager.train_and_evaluate(
                    model, arch, dataset_name, train_dataset, val_dataset, test_dataset, num_classes
                )

                if history is not None:
                    dataset_histories[arch] = history.history
                    # Plot training history after each model completes training
                    visualizer.plot_training_history(history, arch, dataset_name)

                # Save DenseNet121 model trained on 'covid' dataset
                if dataset_name == 'covid' and arch == 'DenseNet121':
                    model.save('densenet121_covid_model.h5')
                    print(f"✅ Saved DenseNet121 model for COVID dataset")

                # Clean up memory
                keras.backend.clear_session()
                del model

            except Exception as e:
                print(f"❌ Error with {arch} on {dataset_name}: {e}")
                keras.backend.clear_session()
                continue

        training_histories[dataset_name] = dataset_histories

    # Collect all results
    all_results = benchmark_manager.results

    if all_results:
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)

        # Save results
        results_df.to_csv('archnet_real_results.csv', index=False)
        print(f"\n💾 Results saved to 'archnet_real_results.csv'")

        # Display results table
        print("\n📋 FINAL RESULTS:")
        print("="*80)
        print(results_df.round(4))

        # Create visualizations
        visualizer.plot_benchmark_results(results_df)

        # Save training histories
        try:
            with open('training_histories.json', 'w') as f:
                json.dump(training_histories, f, indent=2)
            print("💾 Training histories saved to 'training_histories.json'")
        except Exception as e:
            print(f"❌ Error saving training histories: {e}")

        # Performance summary
        print(f"\n🎯 PERFORMANCE SUMMARY:")
        print("="*50)
        for dataset in results_df['dataset_name'].unique():
            dataset_results = results_df[results_df['dataset_name'] == dataset]
            if not dataset_results.empty:
                best_model = dataset_results.loc[dataset_results['test_accuracy'].idxmax()]
                print(f"📁 {dataset.upper()}:")
                print(f"   🏆 Best Model: {best_model['model_name']} "
                      f"(Accuracy: {best_model['test_accuracy']:.3f})")
                print(f"   ⚡ Fastest: {dataset_results.loc[dataset_results['inference_time'].idxmin()]['model_name']}")
                print(f"   📦 Smallest: {dataset_results.loc[dataset_results['model_size_mb'].idxmin()]['model_name']}")
            else:
                print(f"📁 {dataset.upper()}: No results found.")

    else:
        print("❌ No results generated")

    print(f"\n🎉 ArchNet Real Dataset Pipeline Completed!")
    print("="*60)

# =============================================================================
# ADDITIONAL VISUALIZATIONS (Part 2 of your original code, now integrated)
# =============================================================================

def run_additional_visualizations():
    """Generates extra plots after the main pipeline completes and saves results."""
    try:
        # Load the results DataFrame created by main()
        results_df = pd.read_csv('archnet_real_results.csv')

        print("\n🎨 Generating additional visualizations...")
        

        # 1. Heatmap of Test Accuracy by Model and Dataset
        plt.figure(figsize=(10, 6))
        accuracy_pivot = results_df.pivot_table(index='model_name', columns='dataset_name', values='test_accuracy')
        sns.heatmap(accuracy_pivot, annot=True, cmap='viridis', fmt=".3f", linewidths=.5)
        plt.title('Test Accuracy Heatmap by Model and Dataset')
        plt.ylabel('Model Architecture')
        plt.xlabel('Dataset')
        plt.show()

        # 2. Heatmap of F1-Score by Model and Dataset
        plt.figure(figsize=(10, 6))
        f1_pivot = results_df.pivot_table(index='model_name', columns='dataset_name', values='f1_score')
        sns.heatmap(f1_pivot, annot=True, cmap='plasma', fmt=".3f", linewidths=.5)
        plt.title('F1-Score Heatmap by Model and Dataset')
        plt.ylabel('Model Architecture')
        plt.xlabel('Dataset')
        plt.show()

        # 3. Bar plot for inference time comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(x='model_name', y='inference_time', hue='dataset_name', data=results_df, palette='magma')
        plt.title('Inference Time Comparison by Model and Dataset')
        plt.xlabel('Model Architecture')
        plt.ylabel('Inference Time (seconds/image)')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Dataset')
        plt.tight_layout()
        plt.show()

        # 4. Scatter plot: Accuracy vs. Model Size
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='model_size_mb', y='test_accuracy', hue='model_name', style='dataset_name', data=results_df, s=200, alpha=0.7)
        plt.title('Accuracy vs. Model Size (MB)')
        plt.xlabel('Model Size (MB)')
        plt.ylabel('Test Accuracy')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

        print("✅ Additional visualizations generated.")

    except FileNotFoundError:
        print("\n⚠️ Cannot run additional visualizations. Ensure 'archnet_real_results.csv' was created successfully by the main pipeline.")
    except Exception as e:
        print(f"\n❌ An error occurred during additional visualization generation: {e}")

# =============================================================================
# FINAL EXECUTION CALL
# =============================================================================
if __name__ == "__main__":
    main()
    # Run the second part of the script after the main function finishes
    run_additional_visualizations()